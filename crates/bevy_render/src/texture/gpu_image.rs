use crate::{
    render_asset::{
        AssetExtractionError, PrepareAssetError, RenderAsset, RenderAssetBytesPerFrameLimiter,
    },
    render_resource::{DefaultImageSampler, Sampler, Texture, TextureView},
    renderer::{RenderDevice, RenderQueue},
};
use alloc::collections::VecDeque;
use bevy_asset::{AssetId, RenderAssetUsages};
use bevy_ecs::system::{lifetimeless::SRes, Local, SystemParamItem};
use bevy_image::{Image, ImageSampler};
use bevy_log::warn;
use bevy_math::{AspectRatio, UVec2};
use bevy_platform::collections::HashMap;
use core::ops::Range;
use wgpu::{Extent3d, TexelCopyBufferLayout, TextureFormat, TextureUsages};
use wgpu_types::{TextureDescriptor, TextureViewDescriptor};

/// The GPU-representation of an [`Image`].
/// Consists of the [`Texture`], its [`TextureView`] and the corresponding [`Sampler`], and the texture's size.
#[derive(Debug, Clone)]
pub struct GpuImage {
    pub texture: Texture,
    pub texture_view: TextureView,
    pub sampler: Sampler,
    pub texture_descriptor: TextureDescriptor<Option<&'static str>, &'static [TextureFormat]>,
    pub texture_view_descriptor: Option<TextureViewDescriptor<Option<&'static str>>>,
    pub had_data: bool,
}

impl RenderAsset for GpuImage {
    type SourceAsset = Image;
    type Param = (
        SRes<RenderDevice>,
        SRes<RenderQueue>,
        SRes<DefaultImageSampler>,
        SRes<RenderAssetBytesPerFrameLimiter>,
        Local<'static, ImageUploads>,
    );

    #[inline]
    fn asset_usage(image: &Self::SourceAsset) -> RenderAssetUsages {
        image.asset_usage
    }

    fn take_gpu_data(
        source: &mut Self::SourceAsset,
        previous_gpu_asset: Option<&Self>,
    ) -> Result<Self::SourceAsset, AssetExtractionError> {
        let data = source.data.take();

        // check if this image originally had data and no longer does, that implies it
        // has already been extracted
        let valid_upload = data.is_some() || previous_gpu_asset.is_none_or(|prev| !prev.had_data);

        valid_upload
            .then(|| Self::SourceAsset {
                data,
                ..source.clone()
            })
            .ok_or(AssetExtractionError::AlreadyExtracted)
    }

    #[inline]
    fn byte_len(image: &Self::SourceAsset) -> Option<usize> {
        image.data.as_ref().map(Vec::len)
    }

    fn cancel_preparation(
        id: AssetId<Image>,
        (_, _, _, _, uploads): &mut SystemParamItem<Self::Param>,
    ) {
        uploads.0.remove(&id);
    }

    /// Converts the extracted image into a [`GpuImage`].
    fn prepare_asset(
        mut image: Image,
        id: AssetId<Image>,
        (device, queue, sampler, budget, uploads): &mut SystemParamItem<Self::Param>,
        previous: Option<&Self>,
    ) -> Result<Self, PrepareAssetError<Image>> {
        #[cfg(feature = "trace")]
        let _span = bevy_log::info_span!("prepare image", size = ?image.texture_descriptor.size, format = ?image.texture_descriptor.format, bytes = image.data.as_ref().map_or(0, Vec::len)).entered();
        let new_texture = previous.is_none_or(|p| p.texture_descriptor != image.texture_descriptor);
        if !uploads.0.contains_key(&id)
            && new_texture
            && budget.max_assets.is_some()
            && budget
                .max_bytes
                .is_some_and(|max| image.data.as_ref().is_some_and(|data| data.len() > max))
            && image.texture_descriptor.dimension == wgpu::TextureDimension::D2
        {
            let bytes = image.data.take().unwrap();
            let mut empty = image.clone();
            empty.texture_descriptor.usage |= TextureUsages::COPY_DST;
            let mut gpu = Self::from_image(empty, device, queue, sampler, None);
            gpu.texture_descriptor = image.texture_descriptor.clone();
            gpu.had_data = true;
            uploads.0.insert(
                id,
                ImageUpload {
                    gpu,
                    bytes,
                    copies: image_copies(&image.texture_descriptor, image.data_order),
                },
            );
        }
        if let Some(upload) = uploads.0.get_mut(&id) {
            while let Some(copy) = upload.copies.front_mut() {
                let rows = (budget.remaining_bytes() / copy.row as usize)
                    .max(1)
                    .min(copy.range.len() / copy.row as usize) as u32;
                let len = (rows * copy.row) as usize;
                if !budget.try_start(Some(len)) {
                    break;
                }
                queue.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &upload.gpu.texture,
                        mip_level: copy.mip,
                        origin: copy.origin,
                        aspect: wgpu::TextureAspect::All,
                    },
                    &upload.bytes[copy.range.start..copy.range.start + len],
                    TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(copy.row),
                        rows_per_image: None,
                    },
                    Extent3d {
                        height: rows * copy.block_height,
                        ..copy.size
                    },
                );
                budget.write_bytes(len);
                copy.range.start += len;
                copy.origin.y += rows * copy.block_height;
                if copy.range.is_empty() {
                    upload.copies.pop_front();
                }
            }
            if upload.copies.is_empty() {
                return Ok(uploads.0.remove(&id).unwrap().gpu);
            }
            return Err(PrepareAssetError::RetryNextUpdate(image));
        }
        Ok(Self::from_image(image, device, queue, sampler, previous))
    }
}

impl GpuImage {
    fn from_image(
        image: Image,
        render_device: &RenderDevice,
        render_queue: &RenderQueue,
        default_sampler: &DefaultImageSampler,
        previous_asset: Option<&Self>,
    ) -> Self {
        let had_data = image.data.is_some();
        let texture = if let Some(prev) = previous_asset
            && prev.texture_descriptor == image.texture_descriptor
            && (!had_data
                || prev
                    .texture_descriptor
                    .usage
                    .contains(TextureUsages::COPY_DST))
            && let Some(block_bytes) = image.texture_descriptor.format.block_copy_size(None)
        {
            if let Some(ref data) = image.data {
                let (block_width, block_height) =
                    image.texture_descriptor.format.block_dimensions();

                // queue copy
                render_queue.write_texture(
                    prev.texture.as_image_copy(),
                    data,
                    TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(image.width() / block_width * block_bytes),
                        rows_per_image: Some(image.height() / block_height),
                    },
                    image.texture_descriptor.size,
                );
            }

            if !image.copy_on_resize {
                // TODO else could clear here? probably not necessary as textures without data are only useful as render
                // targets and will normally be overwritten immediately anyway
            }

            // reuse previous texture
            prev.texture.clone()
        } else if let Some(ref data) = image.data {
            render_device.create_texture_with_data(
                render_queue,
                &image.texture_descriptor,
                image.data_order,
                data,
            )
        } else {
            let new_texture = render_device.create_texture(&image.texture_descriptor);
            if image.copy_on_resize {
                if let Some(previous) = previous_asset {
                    let mut command_encoder =
                        render_device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("copy_image_on_resize"),
                        });
                    let copy_size = Extent3d {
                        width: image
                            .texture_descriptor
                            .size
                            .width
                            .min(previous.texture_descriptor.size.width),
                        height: image
                            .texture_descriptor
                            .size
                            .height
                            .min(previous.texture_descriptor.size.height),
                        depth_or_array_layers: image
                            .texture_descriptor
                            .size
                            .depth_or_array_layers
                            .min(previous.texture_descriptor.size.depth_or_array_layers),
                    };

                    command_encoder.copy_texture_to_texture(
                        previous.texture.as_image_copy(),
                        new_texture.as_image_copy(),
                        copy_size,
                    );
                    render_queue.submit([command_encoder.finish()]);
                } else {
                    warn!("No previous asset to copy from for image: {:?}", image);
                }
            }
            new_texture
        };

        let texture_view = if let Some(prev) = previous_asset.as_ref()
            && prev.texture_descriptor == image.texture_descriptor
            && prev
                .texture_descriptor
                .usage
                .contains(TextureUsages::COPY_DST)
            && prev.texture_view_descriptor == image.texture_view_descriptor
        {
            prev.texture_view.clone()
        } else {
            image
                .texture_view_descriptor
                .as_ref()
                .map(|desc| texture.create_view(desc))
                .unwrap_or_else(|| texture.create_view(&TextureViewDescriptor::default()))
        };
        let sampler = match image.sampler {
            ImageSampler::Default => default_sampler.0.clone(),
            ImageSampler::Descriptor(descriptor) => {
                render_device.create_sampler(&descriptor.as_wgpu())
            }
        };

        GpuImage {
            texture,
            texture_view,
            sampler,
            texture_descriptor: image.texture_descriptor,
            texture_view_descriptor: image.texture_view_descriptor,
            had_data,
        }
    }
}

impl GpuImage {
    /// Returns the aspect ratio (width / height) of a 2D image.
    #[inline]
    pub fn aspect_ratio(&self) -> AspectRatio {
        AspectRatio::try_from_pixels(
            self.texture_descriptor.size.width,
            self.texture_descriptor.size.height,
        )
        .expect(
            "Failed to calculate aspect ratio: Image dimensions must be positive, non-zero values",
        )
    }

    /// Returns the size of a 2D image.
    #[inline]
    pub fn size_2d(&self) -> UVec2 {
        UVec2::new(
            self.texture_descriptor.size.width,
            self.texture_descriptor.size.height,
        )
    }

    /// Gets the view format of this image.
    /// If the view format is not explicitly provided, falls back to the base image format
    #[inline]
    pub fn view_format(&self) -> TextureFormat {
        self.texture_view_descriptor
            .as_ref()
            .and_then(|view_desc| view_desc.format)
            .unwrap_or(self.texture_descriptor.format)
    }
}

/// Incomplete images stay private until every addressed subresource has been uploaded.
#[derive(Default)]
pub struct ImageUploads(HashMap<AssetId<Image>, ImageUpload>);

struct ImageUpload {
    gpu: GpuImage,
    bytes: Vec<u8>,
    copies: VecDeque<ImageCopy>,
}

struct ImageCopy {
    mip: u32,
    origin: wgpu::Origin3d,
    size: Extent3d,
    row: u32,
    block_height: u32,
    range: Range<usize>,
}

/// The same layer/mip order and physical compressed-block extent as wgpu's whole-image upload.
fn image_copies(
    desc: &TextureDescriptor<Option<&'static str>, &'static [TextureFormat]>,
    order: wgpu::util::TextureDataOrder,
) -> VecDeque<ImageCopy> {
    let (bw, bh) = desc.format.block_dimensions();
    let bytes = desc.format.block_copy_size(None).expect("a color texture");
    let layers = desc.array_layer_count();
    let (outer, inner) = match order {
        wgpu::util::TextureDataOrder::LayerMajor => (layers, desc.mip_level_count),
        wgpu::util::TextureDataOrder::MipMajor => (desc.mip_level_count, layers),
    };
    let (mut copies, mut offset) = (VecDeque::new(), 0);
    for a in 0..outer {
        for b in 0..inner {
            let (layer, mip) = match order {
                wgpu::util::TextureDataOrder::LayerMajor => (a, b),
                wgpu::util::TextureDataOrder::MipMajor => (b, a),
            };
            let mut size = desc.mip_level_size(mip).unwrap().physical_size(desc.format);
            size.depth_or_array_layers = 1;
            let row = size.width / bw * bytes;
            let len = (row * (size.height / bh)) as usize;
            copies.push_back(ImageCopy {
                mip,
                origin: wgpu::Origin3d {
                    x: 0,
                    y: 0,
                    z: layer,
                },
                size,
                row,
                block_height: bh,
                range: offset..offset + len,
            });
            offset += len;
        }
    }
    copies
}

#[cfg(test)]
mod upload_tests {
    use super::*;
    use crate::{
        render_asset::{
            prepare_assets, reset_render_asset_bytes_per_frame, ExtractedAssets,
            PrepareNextFrameAssets, RenderAssets,
        },
        test_utils::create_dummy_device,
    };
    use bevy_app::{App, Update};
    use bevy_asset::Assets;
    use bevy_ecs::schedule::IntoScheduleConfigs;
    use wgpu::{util::TextureDataOrder, TextureDimension};

    fn image(height: u32) -> Image {
        Image::new(
            Extent3d {
                width: 2,
                height,
                depth_or_array_layers: 1,
            },
            TextureDimension::D2,
            (0..2 * height).map(|n| n as u8).collect(),
            TextureFormat::R8Unorm,
            RenderAssetUsages::default(),
        )
    }

    fn app() -> App {
        let (device, queue) = create_dummy_device();
        let sampler = DefaultImageSampler(device.create_sampler(&Default::default()));
        let mut budget = RenderAssetBytesPerFrameLimiter::default();
        budget.max_assets = Some(64);
        budget.max_bytes = Some(4);
        let mut app = App::new();
        app.insert_resource(device)
            .insert_resource(queue)
            .insert_resource(sampler)
            .insert_resource(budget)
            .init_resource::<ExtractedAssets<GpuImage>>()
            .init_resource::<PrepareNextFrameAssets<GpuImage>>()
            .init_resource::<RenderAssets<GpuImage>>()
            .add_systems(
                Update,
                (
                    reset_render_asset_bytes_per_frame,
                    prepare_assets::<GpuImage>,
                )
                    .chain(),
            );
        app
    }

    #[test]
    fn large_images_publish_only_after_all_rows_and_keep_in_place_updates_whole() {
        let mut app = app();
        let mut source = Assets::<Image>::default();
        let handle = source.add(image(6));
        app.world_mut()
            .resource_mut::<ExtractedAssets<GpuImage>>()
            .extracted
            .push((handle.id(), image(6)));
        for _ in 0..2 {
            app.update();
            assert!(app
                .world()
                .resource::<RenderAssets<GpuImage>>()
                .get(handle.id())
                .is_none());
            assert!(app
                .world()
                .resource::<RenderAssetBytesPerFrameLimiter>()
                .has_pending());
            assert_eq!(
                app.world()
                    .resource::<RenderAssetBytesPerFrameLimiter>()
                    .bytes_written
                    .load(core::sync::atomic::Ordering::Relaxed),
                4
            );
        }
        app.update();
        let gpu = app
            .world()
            .resource::<RenderAssets<GpuImage>>()
            .get(handle.id())
            .unwrap();
        let first = gpu.texture.id();
        assert_eq!(gpu.texture_descriptor.size.height, 6);
        assert!(gpu.had_data);
        assert!(!app
            .world()
            .resource::<RenderAssetBytesPerFrameLimiter>()
            .has_pending());
        app.world_mut()
            .resource_mut::<ExtractedAssets<GpuImage>>()
            .extracted
            .push((handle.id(), image(6)));
        app.update();
        assert_eq!(
            app.world()
                .resource::<RenderAssets<GpuImage>>()
                .get(handle.id())
                .unwrap()
                .texture
                .id(),
            first
        );
        assert!(
            !app.world()
                .resource::<RenderAssetBytesPerFrameLimiter>()
                .has_pending(),
            "an in-place update stays whole"
        );
    }

    #[test]
    fn array_subresource_writes_share_the_attempt_budget() {
        let mut app = app();
        {
            let mut budget = app
                .world_mut()
                .resource_mut::<RenderAssetBytesPerFrameLimiter>();
            budget.max_assets = Some(2);
            budget.max_bytes = Some(8);
        }
        let image = Image::new(
            Extent3d {
                width: 2,
                height: 1,
                depth_or_array_layers: 6,
            },
            TextureDimension::D2,
            vec![1; 12],
            TextureFormat::R8Unorm,
            RenderAssetUsages::default(),
        );
        let mut source = Assets::<Image>::default();
        let handle = source.add(image.clone());
        app.world_mut()
            .resource_mut::<ExtractedAssets<GpuImage>>()
            .extracted
            .push((handle.id(), image));
        for _ in 0..5 {
            app.update();
            assert!(app
                .world()
                .resource::<RenderAssets<GpuImage>>()
                .get(handle.id())
                .is_none());
            assert_eq!(
                app.world()
                    .resource::<RenderAssetBytesPerFrameLimiter>()
                    .bytes_written
                    .load(core::sync::atomic::Ordering::Relaxed),
                2
            );
        }
        app.update();
        assert_eq!(
            app.world()
                .resource::<RenderAssets<GpuImage>>()
                .get(handle.id())
                .unwrap()
                .texture_descriptor
                .size
                .depth_or_array_layers,
            6
        );
        assert!(!app
            .world()
            .resource::<RenderAssetBytesPerFrameLimiter>()
            .has_pending());
    }

    #[test]
    fn replaced_and_removed_images_discard_unpublished_uploads() {
        let mut app = app();
        let mut source = Assets::<Image>::default();
        let handle = source.add(image(6));
        app.world_mut()
            .resource_mut::<ExtractedAssets<GpuImage>>()
            .extracted
            .push((handle.id(), image(6)));
        app.update();
        {
            let mut extracted = app.world_mut().resource_mut::<ExtractedAssets<GpuImage>>();
            extracted.added.insert(handle.id());
            extracted.extracted.push((handle.id(), image(4)));
        }
        app.update();
        app.world_mut()
            .resource_mut::<ExtractedAssets<GpuImage>>()
            .added
            .clear();
        assert!(app
            .world()
            .resource::<RenderAssets<GpuImage>>()
            .get(handle.id())
            .is_none());
        app.update();
        assert_eq!(
            app.world()
                .resource::<RenderAssets<GpuImage>>()
                .get(handle.id())
                .unwrap()
                .texture_descriptor
                .size
                .height,
            4
        );
        {
            let mut extracted = app.world_mut().resource_mut::<ExtractedAssets<GpuImage>>();
            extracted.added.insert(handle.id());
            extracted.extracted.push((handle.id(), image(8)));
        }
        app.update();
        app.world_mut()
            .resource_mut::<ExtractedAssets<GpuImage>>()
            .added
            .clear();
        app.world_mut()
            .resource_mut::<ExtractedAssets<GpuImage>>()
            .removed
            .insert(handle.id());
        app.update();
        assert!(app
            .world()
            .resource::<RenderAssets<GpuImage>>()
            .get(handle.id())
            .is_none());
        assert!(!app
            .world()
            .resource::<RenderAssetBytesPerFrameLimiter>()
            .has_pending());
    }

    #[test]
    fn compressed_mips_keep_both_data_orders_and_physical_blocks() {
        let mut descriptor = image(8).texture_descriptor;
        descriptor.size = Extent3d {
            width: 8,
            height: 8,
            depth_or_array_layers: 2,
        };
        descriptor.format = TextureFormat::Bc1RgbaUnorm;
        descriptor.mip_level_count = 3;
        let copies = |order| {
            image_copies(&descriptor, order)
                .into_iter()
                .map(|c| (c.origin.z, c.mip, c.size.width, c.size.height, c.range))
                .collect::<Vec<_>>()
        };
        assert_eq!(
            copies(TextureDataOrder::LayerMajor),
            [
                (0, 0, 8, 8, 0..32),
                (0, 1, 4, 4, 32..40),
                (0, 2, 4, 4, 40..48),
                (1, 0, 8, 8, 48..80),
                (1, 1, 4, 4, 80..88),
                (1, 2, 4, 4, 88..96)
            ]
        );
        assert_eq!(
            copies(TextureDataOrder::MipMajor),
            [
                (0, 0, 8, 8, 0..32),
                (1, 0, 8, 8, 32..64),
                (0, 1, 4, 4, 64..72),
                (1, 1, 4, 4, 72..80),
                (0, 2, 4, 4, 80..88),
                (1, 2, 4, 4, 88..96)
            ]
        );
    }
}
