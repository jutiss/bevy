use crate::{
    render_resource::AsBindGroupError, Extract, ExtractSchedule, MainWorld, Render, RenderApp,
    RenderStartup, RenderSystems, Res,
};
use bevy_app::{App, Plugin, SubApp};
use bevy_asset::{Asset, AssetEvent, AssetId, Assets, RenderAssetUsages};
use bevy_ecs::{
    prelude::{Commands, IntoScheduleConfigs, Local, MessageReader, ResMut, Resource},
    schedule::{ScheduleConfigs, SystemSet},
    system::{ScheduleSystem, StaticSystemParam, SystemParam, SystemParamItem, SystemState},
    world::{FromWorld, Mut},
};
use bevy_log::{debug, error};
use bevy_platform::collections::{HashMap, HashSet};
use core::marker::PhantomData;
use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum PrepareAssetError<E: Send + Sync + 'static> {
    #[error("Failed to prepare asset")]
    RetryNextUpdate(E),
    #[error("Failed to build bind group: {0}")]
    AsBindGroupError(AsBindGroupError),
}

/// The system set during which we extract modified assets to the render world.
#[derive(SystemSet, Clone, PartialEq, Eq, Debug, Hash)]
pub struct AssetExtractionSystems;

/// Error returned when an asset due for extraction has already been extracted
#[derive(Debug, Error)]
pub enum AssetExtractionError {
    #[error("The asset has already been extracted")]
    AlreadyExtracted,
    #[error("The asset type does not support extraction. To clone the asset to the renderworld, use `RenderAssetUsages::default()`")]
    NoExtractionImplementation,
}

/// Describes how an asset gets extracted and prepared for rendering.
///
/// In the [`ExtractSchedule`] step the [`RenderAsset::SourceAsset`] is transferred
/// from the "main world" into the "render world".
///
/// After that in the [`RenderSystems::PrepareAssets`] step the extracted asset
/// is transformed into its GPU-representation of type [`RenderAsset`].
pub trait RenderAsset: Send + Sync + 'static + Sized {
    /// The representation of the asset in the "main world".
    type SourceAsset: Asset + Clone;

    /// Specifies all ECS data required by [`RenderAsset::prepare_asset`].
    ///
    /// For convenience use the [`lifetimeless`](bevy_ecs::system::lifetimeless) [`SystemParam`].
    type Param: SystemParam;

    /// Whether or not to unload the asset after extracting it to the render world.
    #[inline]
    fn asset_usage(_source_asset: &Self::SourceAsset) -> RenderAssetUsages {
        RenderAssetUsages::default()
    }

    /// Size of the data the asset will upload to the gpu. Specifying a return value
    /// will allow the asset to be throttled via [`RenderAssetBytesPerFrame`].
    #[inline]
    #[expect(
        unused_variables,
        reason = "The parameters here are intentionally unused by the default implementation; however, putting underscores here will result in the underscores being copied by rust-analyzer's tab completion."
    )]
    fn byte_len(source_asset: &Self::SourceAsset) -> Option<usize> {
        None
    }

    /// Prepares the [`RenderAsset::SourceAsset`] for the GPU by transforming it into a [`RenderAsset`].
    ///
    /// ECS data may be accessed via `param`.
    fn prepare_asset(
        source_asset: Self::SourceAsset,
        asset_id: AssetId<Self::SourceAsset>,
        param: &mut SystemParamItem<Self::Param>,
        previous_asset: Option<&Self>,
    ) -> Result<Self, PrepareAssetError<Self::SourceAsset>>;

    /// Called whenever the [`RenderAsset::SourceAsset`] has been removed.
    ///
    /// You can implement this method if you need to access ECS data (via
    /// `_param`) in order to perform cleanup tasks when the asset is removed.
    ///
    /// The default implementation does nothing.
    fn unload_asset(
        _source_asset: AssetId<Self::SourceAsset>,
        _param: &mut SystemParamItem<Self::Param>,
    ) {
    }

    /// Keep the last complete GPU representation while a replacement is preparing.
    fn retain_previous() -> bool {
        false
    }

    /// Discard preparation retained for a source that was replaced or removed.
    fn cancel_preparation(
        _source_asset: AssetId<Self::SourceAsset>,
        _param: &mut SystemParamItem<Self::Param>,
    ) {
    }

    /// Make a copy of the asset to be moved to the `RenderWorld` / gpu. Heavy internal data (pixels, vertex attributes)
    /// should be moved into the copy, leaving this asset with only metadata.
    /// An error may be returned to indicate that the asset has already been extracted, and should not
    /// have been modified on the CPU side (as it cannot be transferred to GPU again).
    /// The previous GPU asset is also provided, which can be used to check if the modification is valid.
    fn take_gpu_data(
        _source: &mut Self::SourceAsset,
        _previous_gpu_asset: Option<&Self>,
    ) -> Result<Self::SourceAsset, AssetExtractionError> {
        Err(AssetExtractionError::NoExtractionImplementation)
    }
}

/// This plugin extracts the changed assets from the "app world" into the "render world"
/// and prepares them for the GPU. They can then be accessed from the [`RenderAssets`] resource.
///
/// Therefore it sets up the [`ExtractSchedule`] and
/// [`RenderSystems::PrepareAssets`] steps for the specified [`RenderAsset`].
///
/// The `AFTER` generic parameter can be used to specify that `A::prepare_asset` should not be run until
/// `prepare_assets::<AFTER>` has completed. This allows the `prepare_asset` function to depend on another
/// prepared [`RenderAsset`], for example `Mesh::prepare_asset` relies on `RenderAssets::<GpuImage>` for morph
/// targets, so the plugin is created as `RenderAssetPlugin::<RenderMesh, GpuImage>::default()`.
pub struct RenderAssetPlugin<A: RenderAsset, AFTER: RenderAssetDependency + 'static = ()> {
    phantom: PhantomData<fn() -> (A, AFTER)>,
}

impl<A: RenderAsset, AFTER: RenderAssetDependency + 'static> Default
    for RenderAssetPlugin<A, AFTER>
{
    fn default() -> Self {
        Self {
            phantom: Default::default(),
        }
    }
}

impl<A: RenderAsset, AFTER: RenderAssetDependency + 'static> Plugin
    for RenderAssetPlugin<A, AFTER>
{
    fn build(&self, app: &mut App) {
        app.init_resource::<CachedExtractRenderAssetSystemState<A>>();
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .init_resource::<ExtractedAssets<A>>()
                .init_resource::<RenderAssets<A>>()
                .allow_ambiguous_resource::<RenderAssets<A>>()
                .init_resource::<PrepareNextFrameAssets<A>>()
                .add_systems(RenderStartup, collect_render_assets_to_reextract::<A>)
                .add_systems(
                    ExtractSchedule,
                    extract_render_asset::<A>.in_set(AssetExtractionSystems),
                );
            AFTER::register_system(
                render_app,
                prepare_assets::<A>.in_set(RenderSystems::PrepareAssets),
            );
        }
    }
}

// helper to allow specifying dependencies between render assets
pub trait RenderAssetDependency {
    fn register_system(render_app: &mut SubApp, system: ScheduleConfigs<ScheduleSystem>);
}

impl RenderAssetDependency for () {
    fn register_system(render_app: &mut SubApp, system: ScheduleConfigs<ScheduleSystem>) {
        render_app.add_systems(Render, system);
    }
}

impl<A: RenderAsset> RenderAssetDependency for A {
    fn register_system(render_app: &mut SubApp, system: ScheduleConfigs<ScheduleSystem>) {
        render_app.add_systems(Render, system.after(prepare_assets::<A>));
    }
}

impl<A, B> RenderAssetDependency for (A, B)
where
    A: RenderAsset,
    B: RenderAsset,
{
    fn register_system(render_app: &mut SubApp, system: ScheduleConfigs<ScheduleSystem>) {
        render_app.add_systems(
            Render,
            system.after(prepare_assets::<A>).after(prepare_assets::<B>),
        );
    }
}

/// Temporarily stores the extracted and removed assets of the current frame.
#[derive(Resource)]
pub struct ExtractedAssets<A: RenderAsset> {
    /// The assets extracted this frame.
    ///
    /// These are assets that were either added or modified this frame.
    pub extracted: Vec<(AssetId<A::SourceAsset>, A::SourceAsset)>,

    /// IDs of the assets that were removed this frame.
    ///
    /// These assets will not be present in [`ExtractedAssets::extracted`].
    pub removed: HashSet<AssetId<A::SourceAsset>>,

    /// IDs of the assets that were modified this frame.
    pub modified: HashSet<AssetId<A::SourceAsset>>,

    /// IDs of the assets that were added this frame.
    pub added: HashSet<AssetId<A::SourceAsset>>,
}

impl<A: RenderAsset> Default for ExtractedAssets<A> {
    fn default() -> Self {
        Self {
            extracted: Default::default(),
            removed: Default::default(),
            modified: Default::default(),
            added: Default::default(),
        }
    }
}

/// Stores all GPU representations ([`RenderAsset`])
/// of [`RenderAsset::SourceAsset`] as long as they exist.
#[derive(Resource)]
pub struct RenderAssets<A: RenderAsset>(HashMap<AssetId<A::SourceAsset>, A>);

impl<A: RenderAsset> Default for RenderAssets<A> {
    fn default() -> Self {
        Self(Default::default())
    }
}

impl<A: RenderAsset> RenderAssets<A> {
    pub fn get(&self, id: impl Into<AssetId<A::SourceAsset>>) -> Option<&A> {
        self.0.get(&id.into())
    }

    pub fn get_mut(&mut self, id: impl Into<AssetId<A::SourceAsset>>) -> Option<&mut A> {
        self.0.get_mut(&id.into())
    }

    pub fn insert(&mut self, id: impl Into<AssetId<A::SourceAsset>>, value: A) -> Option<A> {
        self.0.insert(id.into(), value)
    }

    pub fn remove(&mut self, id: impl Into<AssetId<A::SourceAsset>>) -> Option<A> {
        self.0.remove(&id.into())
    }

    pub fn iter(&self) -> impl Iterator<Item = (AssetId<A::SourceAsset>, &A)> {
        self.0.iter().map(|(k, v)| (*k, v))
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = (AssetId<A::SourceAsset>, &mut A)> {
        self.0.iter_mut().map(|(k, v)| (*k, v))
    }
}

#[derive(Resource)]
struct CachedExtractRenderAssetSystemState<A: RenderAsset> {
    state: SystemState<(
        MessageReader<'static, 'static, AssetEvent<A::SourceAsset>>,
        ResMut<'static, Assets<A::SourceAsset>>,
        Option<Res<'static, RenderAssets<A>>>,
    )>,
}

impl<A: RenderAsset> FromWorld for CachedExtractRenderAssetSystemState<A> {
    fn from_world(world: &mut bevy_ecs::world::World) -> Self {
        Self {
            state: SystemState::new(world),
        }
    }
}

/// Resource inserted during [`RenderStartup`] containing asset IDs that need
/// re-extraction from the main world after device recovery.
#[derive(Resource)]
pub(crate) struct RenderAssetsToReExtract<A: RenderAsset> {
    ids: Vec<AssetId<A::SourceAsset>>,
}

/// Drains all asset IDs from [`RenderAssets<A>`] to mark for re-extraction.
fn collect_render_assets_to_reextract<A: RenderAsset>(
    mut commands: Commands,
    mut render_assets: ResMut<RenderAssets<A>>,
    mut prepare_next_frame: ResMut<PrepareNextFrameAssets<A>>,
) {
    let ids: Vec<_> = render_assets.0.drain().map(|(id, _)| id).collect();
    prepare_next_frame.assets.clear();
    if !ids.is_empty() {
        commands.insert_resource(RenderAssetsToReExtract::<A> { ids });
    }
}

/// Extracts all created or modified assets of the corresponding [`RenderAsset::SourceAsset`] type
/// into the "render world", including any assets invalidated by device recovery.
pub(crate) fn extract_render_asset<A: RenderAsset>(
    mut to_reextract: Option<ResMut<RenderAssetsToReExtract<A>>>,
    mut extracted_assets: ResMut<ExtractedAssets<A>>,
    mut main_world: ResMut<MainWorld>,
    mut needs_extracting: Local<HashSet<AssetId<A::SourceAsset>>>,
) {
    extracted_assets.extracted.clear();
    extracted_assets.removed.clear();
    extracted_assets.modified.clear();
    extracted_assets.added.clear();
    needs_extracting.clear();

    let reextract_ids = to_reextract
        .as_mut()
        .map(|r| core::mem::take(&mut r.ids))
        .filter(|ids| !ids.is_empty());

    main_world.resource_scope(
        |world, mut cached_state: Mut<CachedExtractRenderAssetSystemState<A>>| {
            let (mut events, mut assets, maybe_render_assets) = cached_state.state.get_mut(world).unwrap();

            if let Some(reextract_ids) = reextract_ids {
                needs_extracting.extend(reextract_ids);
            }

            for event in events.read() {
                #[expect(
                    clippy::match_same_arms,
                    reason = "LoadedWithDependencies is marked as a TODO, so it's likely this will no longer lint soon."
                )]
                match event {
                    AssetEvent::Added { id } => {
                        needs_extracting.insert(*id);
                    }
                    AssetEvent::Modified { id } => {
                        needs_extracting.insert(*id);
                        extracted_assets.modified.insert(*id);
                    }
                    AssetEvent::Removed { .. } => {
                        // We don't care that the asset was removed from Assets<T> in the main world.
                        // An asset is only removed from RenderAssets<T> when its last handle is dropped (AssetEvent::Unused).
                    }
                    AssetEvent::Unused { id } => {
                        needs_extracting.remove(id);
                        extracted_assets.modified.remove(id);
                        extracted_assets.removed.insert(*id);
                    }
                    AssetEvent::LoadedWithDependencies { .. } => {
                        // TODO: handle this
                    }
                }
            }

            for id in needs_extracting.drain() {
                if let Some(asset) = assets.get(id) {
                    let asset_usage = A::asset_usage(asset);
                    if asset_usage.contains(RenderAssetUsages::RENDER_WORLD) {
                        if asset_usage == RenderAssetUsages::RENDER_WORLD {
                            if let Some(asset) = assets.get_mut_untracked(id) {
                                let previous_asset = maybe_render_assets.as_ref().and_then(|render_assets| render_assets.get(id));
                                match A::take_gpu_data(asset, previous_asset) {
                                    Ok(gpu_data_asset) => {
                                        extracted_assets.extracted.push((id, gpu_data_asset));
                                        extracted_assets.added.insert(id);
                                    }
                                    Err(e) => {
                                        error!("{} with RenderAssetUsages == RENDER_WORLD cannot be extracted: {e}", core::any::type_name::<A>());
                                    }
                                };
                            }
                        } else {
                            extracted_assets.extracted.push((id, asset.clone()));
                            extracted_assets.added.insert(id);
                        }
                    }
                }
            }

            cached_state.state.apply(world);
        },
    );
}

// TODO: consider storing inside system?
/// All assets that should be prepared next frame.
#[derive(Resource)]
pub struct PrepareNextFrameAssets<A: RenderAsset> {
    assets: Vec<(AssetId<A::SourceAsset>, A::SourceAsset)>,
}

impl<A: RenderAsset> PrepareNextFrameAssets<A> {
    /// Whether a source still has a replacement or first representation preparing.
    pub fn contains(&self, id: AssetId<A::SourceAsset>) -> bool {
        self.assets.iter().any(|(pending, _)| *pending == id)
    }
}

impl<A: RenderAsset> Default for PrepareNextFrameAssets<A> {
    fn default() -> Self {
        Self {
            assets: Default::default(),
        }
    }
}

/// This system prepares all assets of the corresponding [`RenderAsset::SourceAsset`] type
/// which where extracted this frame for the GPU.
pub fn prepare_assets<A: RenderAsset>(
    mut extracted_assets: ResMut<ExtractedAssets<A>>,
    mut render_assets: ResMut<RenderAssets<A>>,
    mut prepare_next_frame: ResMut<PrepareNextFrameAssets<A>>,
    param: StaticSystemParam<<A as RenderAsset>::Param>,
    bpf: Res<RenderAssetBytesPerFrameLimiter>,
) {
    let mut wrote_asset_count = 0;

    let mut param = param.into_inner();
    for id in extracted_assets.removed.union(&extracted_assets.added) {
        A::cancel_preparation(*id, &mut param);
    }
    let queued_assets = core::mem::take(&mut prepare_next_frame.assets);
    for (id, extracted_asset) in queued_assets {
        if extracted_assets.removed.contains(&id) || extracted_assets.added.contains(&id) {
            // skip previous frame's assets that have been removed or updated
            continue;
        }

        let size = A::byte_len(&extracted_asset);
        if render_assets.get(id).is_none() && !bpf.try_start(size) {
            prepare_next_frame.assets.push((id, extracted_asset));
            continue;
        }
        let write_bytes = size.unwrap_or(0);

        let previous_asset = render_assets.get(id);
        match A::prepare_asset(extracted_asset, id, &mut param, previous_asset) {
            Ok(prepared_asset) => {
                render_assets.insert(id, prepared_asset);
                bpf.write_bytes(write_bytes);
                wrote_asset_count += 1;
            }
            Err(PrepareAssetError::RetryNextUpdate(extracted_asset)) => {
                bpf.defer();
                prepare_next_frame.assets.push((id, extracted_asset));
            }
            Err(PrepareAssetError::AsBindGroupError(e)) => {
                error!(
                    "{} Bind group construction failed: {e}",
                    core::any::type_name::<A>()
                );
            }
        }
    }

    for removed in extracted_assets.removed.drain() {
        render_assets.remove(removed);
        A::unload_asset(removed, &mut param);
    }

    for (id, extracted_asset) in extracted_assets.extracted.drain(..) {
        // Assets may opt to retain their last complete representation until replacement succeeds.
        let previous_asset = render_assets.remove(id);

        let size = A::byte_len(&extracted_asset);
        if previous_asset.is_none() && !bpf.try_start(size) {
            prepare_next_frame.assets.push((id, extracted_asset));
            continue;
        }
        let write_bytes = size.unwrap_or(0);

        match A::prepare_asset(extracted_asset, id, &mut param, previous_asset.as_ref()) {
            Ok(prepared_asset) => {
                render_assets.insert(id, prepared_asset);
                bpf.write_bytes(write_bytes);
                wrote_asset_count += 1;
            }
            Err(PrepareAssetError::RetryNextUpdate(extracted_asset)) => {
                bpf.defer();
                prepare_next_frame.assets.push((id, extracted_asset));
                if A::retain_previous()
                    && let Some(previous) = previous_asset
                {
                    render_assets.insert(id, previous);
                }
            }
            Err(PrepareAssetError::AsBindGroupError(e)) => {
                error!(
                    "{} Bind group construction failed: {e}",
                    core::any::type_name::<A>()
                );
            }
        }
    }

    if bpf.exhausted() && !prepare_next_frame.assets.is_empty() {
        debug!(
            "{} write budget exhausted with {} assets remaining (wrote {})",
            core::any::type_name::<A>(),
            prepare_next_frame.assets.len(),
            wrote_asset_count
        );
    }
}

pub fn reset_render_asset_bytes_per_frame(
    mut bpf_limiter: ResMut<RenderAssetBytesPerFrameLimiter>,
) {
    bpf_limiter.reset();
}

pub fn extract_render_asset_bytes_per_frame(
    bpf: Extract<Res<RenderAssetBytesPerFrame>>,
    mut bpf_limiter: ResMut<RenderAssetBytesPerFrameLimiter>,
) {
    bpf_limiter.max_bytes = bpf.max_bytes;
    bpf_limiter.max_assets = bpf.max_assets;
    bpf_limiter.limit_specializations = bpf.limit_specializations;
}

/// A resource that defines the amount of data allowed to be transferred from CPU to GPU
/// each frame, preventing choppy frames at the cost of waiting longer for GPU assets
/// to become available.
#[derive(Resource, Default)]
pub struct RenderAssetBytesPerFrame {
    pub max_bytes: Option<usize>,
    /// Maximum initial preparation attempts, including retries and unsized assets; existing updates stay in-frame.
    pub max_assets: Option<usize>,
    /// Share the attempt limit with uncached mesh specializations during cold preparation.
    pub limit_specializations: bool,
}

impl RenderAssetBytesPerFrame {
    /// `max_bytes`: the number of bytes to write per frame.
    ///
    /// This is a soft limit: only full assets are written currently, uploading stops
    /// after the first asset that exceeds the limit.
    ///
    /// To participate, assets should implement [`RenderAsset::byte_len`]. If the default
    /// is not overridden, the assets are assumed to be small enough to upload without restriction.
    pub fn new(max_bytes: usize) -> Self {
        Self {
            max_bytes: Some(max_bytes),
            max_assets: None,
            limit_specializations: false,
        }
    }
}

/// A render-world resource that facilitates limiting the data transferred from CPU to GPU
/// each frame, preventing choppy frames at the cost of waiting longer for GPU assets
/// to become available.
#[derive(Resource, Default)]
pub struct RenderAssetBytesPerFrameLimiter {
    /// Populated by [`RenderAssetBytesPerFrame`] during extraction.
    pub max_bytes: Option<usize>,
    /// Bytes written this frame.
    pub bytes_written: AtomicUsize,
    /// Maximum preparation attempts, copied during extraction.
    pub max_assets: Option<usize>,
    /// Whether uncached mesh specializations share this frame's attempt limit.
    pub limit_specializations: bool,
    attempts: AtomicUsize,
    pending: AtomicBool,
}

impl RenderAssetBytesPerFrameLimiter {
    /// Reset the available bytes. Called once per frame during extraction by [`crate::RenderPlugin`].
    pub fn reset(&mut self) {
        self.bytes_written.store(0, Ordering::Relaxed);
        self.attempts.store(0, Ordering::Relaxed);
        self.pending.store(false, Ordering::Relaxed);
    }

    /// Whether preparation deferred any asset this frame, either for its budget or dependencies.
    pub fn has_pending(&self) -> bool {
        self.pending.load(Ordering::Relaxed)
    }

    pub(crate) fn defer(&self) {
        self.pending.store(true, Ordering::Relaxed);
    }

    /// Admit an asset or unsized preparation item, marking denied work pending.
    pub fn try_start(&self, size: Option<usize>) -> bool {
        if size.is_some() && self.exhausted() {
            self.defer();
            return false;
        }
        if let Some(max) = self.max_assets
            && self.attempts.fetch_add(1, Ordering::Relaxed) >= max
        {
            self.defer();
            return false;
        }
        true
    }

    pub(crate) fn remaining_bytes(&self) -> usize {
        self.max_bytes.map_or(usize::MAX, |max| {
            max.saturating_sub(self.bytes_written.load(Ordering::Relaxed))
        })
    }

    /// Check how many bytes are available for writing.
    pub fn available_bytes(&self, required_bytes: usize) -> usize {
        if let Some(max_bytes) = self.max_bytes {
            let total_bytes = self
                .bytes_written
                .fetch_add(required_bytes, Ordering::Relaxed);

            // The bytes available is the inverse of the amount we overshot max_bytes
            if total_bytes >= max_bytes {
                required_bytes.saturating_sub(total_bytes - max_bytes)
            } else {
                required_bytes
            }
        } else {
            required_bytes
        }
    }

    /// Decreases the available bytes for the current frame.
    pub(crate) fn write_bytes(&self, bytes: usize) {
        if self.max_bytes.is_some() && bytes > 0 {
            self.bytes_written.fetch_add(bytes, Ordering::Relaxed);
        }
    }

    /// Returns `true` if there are no remaining bytes available for writing this frame.
    pub(crate) fn exhausted(&self) -> bool {
        if let Some(max_bytes) = self.max_bytes {
            let bytes_written = self.bytes_written.load(Ordering::Relaxed);
            bytes_written >= max_bytes
        } else {
            false
        }
    }
}

#[cfg(test)]
mod budget_tests {
    use super::*;
    use bevy_app::Update;
    use bevy_reflect::TypePath;

    #[derive(Asset, TypePath, Clone)]
    struct Source {
        value: usize,
        retry: bool,
    }
    struct Prepared(usize);
    impl RenderAsset for Prepared {
        type SourceAsset = Source;

        type Param = ();
        fn prepare_asset(
            source: Source,
            _: AssetId<Source>,
            _: &mut SystemParamItem<Self::Param>,
            _: Option<&Self>,
        ) -> Result<Self, PrepareAssetError<Source>> {
            if source.retry {
                Err(PrepareAssetError::RetryNextUpdate(source))
            } else {
                Ok(Self(source.value))
            }
        }
    }

    #[test]
    fn preparation_limit_is_shared_with_unsized_assets_and_resets_each_frame() {
        let mut limit = RenderAssetBytesPerFrameLimiter {
            max_bytes: Some(4),
            max_assets: Some(2),
            ..Default::default()
        };
        assert!(limit.try_start(Some(8)), "one oversized asset progresses");
        limit.write_bytes(8);
        assert!(!limit.try_start(Some(1)), "byte budget exhausted");
        assert!(
            limit.try_start(None),
            "unsized assets use the attempt budget alone"
        );
        assert!(!limit.try_start(None), "attempt budget exhausted");
        assert!(limit.has_pending());
        limit.reset();
        assert!(!limit.has_pending());
        assert!(limit.try_start(None));
        limit.max_assets = None;
        for _ in 0..5 {
            assert!(limit.try_start(None));
        }
        limit.max_assets = Some(0);
        assert!(!limit.try_start(None));
    }

    #[test]
    fn existing_updates_stay_in_frame_when_the_initial_preparation_budget_is_empty() {
        let mut app = App::new();
        let mut limit = RenderAssetBytesPerFrameLimiter::default();
        limit.max_assets = Some(0);
        app.init_resource::<ExtractedAssets<Prepared>>()
            .init_resource::<PrepareNextFrameAssets<Prepared>>()
            .init_resource::<RenderAssets<Prepared>>()
            .insert_resource(limit)
            .add_systems(Update, prepare_assets::<Prepared>);
        let mut sources = Assets::<Source>::default();
        let old = sources.add(Source {
            value: 1,
            retry: false,
        });
        let fresh = sources.add(Source {
            value: 2,
            retry: false,
        });
        app.world_mut()
            .resource_mut::<RenderAssets<Prepared>>()
            .insert(old.id(), Prepared(1));
        app.world_mut()
            .resource_mut::<ExtractedAssets<Prepared>>()
            .extracted = vec![
            (
                old.id(),
                Source {
                    value: 9,
                    retry: false,
                },
            ),
            (
                fresh.id(),
                Source {
                    value: 2,
                    retry: false,
                },
            ),
        ];
        app.update();
        assert_eq!(
            app.world()
                .resource::<RenderAssets<Prepared>>()
                .get(old.id())
                .unwrap()
                .0,
            9
        );
        assert!(app
            .world()
            .resource::<RenderAssets<Prepared>>()
            .get(fresh.id())
            .is_none());
        assert_eq!(
            app.world()
                .resource::<PrepareNextFrameAssets<Prepared>>()
                .assets
                .len(),
            1
        );
    }

    #[test]
    fn preparation_budget_bounds_unsized_retries_and_discards_replaced_or_removed_work() {
        let mut app = App::new();
        app.init_resource::<ExtractedAssets<Prepared>>()
            .init_resource::<PrepareNextFrameAssets<Prepared>>()
            .init_resource::<RenderAssets<Prepared>>()
            .insert_resource(RenderAssetBytesPerFrameLimiter {
                max_assets: Some(2),
                ..Default::default()
            })
            .add_systems(
                Update,
                (
                    reset_render_asset_bytes_per_frame,
                    prepare_assets::<Prepared>,
                )
                    .chain(),
            );
        let mut sources = Assets::<Source>::default();
        let handles: Vec<_> = (0..3)
            .map(|value| {
                sources.add(Source {
                    value,
                    retry: value == 0,
                })
            })
            .collect();
        app.world_mut()
            .resource_mut::<ExtractedAssets<Prepared>>()
            .extracted = handles
            .iter()
            .map(|h| (h.id(), sources.get(h).unwrap().clone()))
            .collect();
        app.update();
        assert_eq!(
            app.world()
                .resource::<RenderAssets<Prepared>>()
                .get(handles[1].id())
                .unwrap()
                .0,
            1
        );
        assert_eq!(
            app.world()
                .resource::<PrepareNextFrameAssets<Prepared>>()
                .assets
                .len(),
            2
        );
        assert!(app
            .world()
            .resource::<RenderAssetBytesPerFrameLimiter>()
            .has_pending());
        {
            let mut extracted = app.world_mut().resource_mut::<ExtractedAssets<Prepared>>();
            extracted.removed.insert(handles[0].id());
            extracted.added.insert(handles[2].id());
            extracted.extracted.push((
                handles[2].id(),
                Source {
                    value: 9,
                    retry: false,
                },
            ));
        }
        app.update();
        assert_eq!(
            app.world()
                .resource::<RenderAssets<Prepared>>()
                .get(handles[2].id())
                .unwrap()
                .0,
            9
        );
        assert!(app
            .world()
            .resource::<PrepareNextFrameAssets<Prepared>>()
            .assets
            .is_empty());
        assert!(!app
            .world()
            .resource::<RenderAssetBytesPerFrameLimiter>()
            .has_pending());
    }
}
