//! Fog formulas adapted from:
//! https://learn.microsoft.com/en-us/windows/win32/direct3d9/fog-formulas
//! https://catlikecoding.com/unity/tutorials/rendering/part-14/
//! https://iquilezles.org/articles/fog/ (Atmospheric Fog and Scattering)

import package::render::{
    mesh_view_bindings::fog,
    mesh_view_types::Fog,
};


fn scattering_adjusted_fog_color(
    fog_params: Fog,
    scattering: vec3<f32>,
) -> vec4<f32> {
    if (fog_params.directional_light_color.a > 0.0) {
        return vec4<f32>(
            fog_params.base_color.rgb
                + scattering * fog_params.directional_light_color.rgb * fog_params.directional_light_color.a,
            fog_params.base_color.a,
        );
    } else {
        return fog_params.base_color;
    }
}

// The linear mode's fog amount at a distance: the share of the way from start to end, saturated, raised to the power (1 for Linear).
fn linear_fog_amount(fog_params: Fog, distance: f32) -> f32 {
    return pow(saturate((distance - fog_params.be.x) / (fog_params.be.y - fog_params.be.x)), fog_params.be.z);
}

fn linear_fog(
    fog_params: Fog,
    input_color: vec4<f32>,
    distance: f32,
    scattering: vec3<f32>,
) -> vec4<f32> {
    var fog_color = scattering_adjusted_fog_color(fog_params, scattering);
    fog_color.a *= linear_fog_amount(fog_params, distance);
    return vec4<f32>(mix(input_color.rgb, fog_color.rgb, fog_color.a), input_color.a);
}

fn exponential_fog(
    fog_params: Fog,
    input_color: vec4<f32>,
    distance: f32,
    scattering: vec3<f32>,
) -> vec4<f32> {
    var fog_color = scattering_adjusted_fog_color(fog_params, scattering);
    let density = fog_params.be.x;
    fog_color.a *= 1.0 - 1.0 / exp(distance * density);
    return vec4<f32>(mix(input_color.rgb, fog_color.rgb, fog_color.a), input_color.a);
}

fn exponential_squared_fog(
    fog_params: Fog,
    input_color: vec4<f32>,
    distance: f32,
    scattering: vec3<f32>,
) -> vec4<f32> {
    var fog_color = scattering_adjusted_fog_color(fog_params, scattering);
    let distance_times_density = distance * fog_params.be.x;
    fog_color.a *= 1.0 - 1.0 / exp(distance_times_density * distance_times_density);
    return vec4<f32>(mix(input_color.rgb, fog_color.rgb, fog_color.a), input_color.a);
}

fn atmospheric_fog(
    fog_params: Fog,
    input_color: vec4<f32>,
    distance: f32,
    scattering: vec3<f32>,
) -> vec4<f32> {
    var fog_color = scattering_adjusted_fog_color(fog_params, scattering);
    let extinction_factor = 1.0 - 1.0 / exp(distance * fog_params.be);
    let inscattering_factor = 1.0 - 1.0 / exp(distance * fog_params.bi);
    return vec4<f32>(
        input_color.rgb * (1.0 - extinction_factor * fog_color.a)
            + fog_color.rgb * inscattering_factor * fog_color.a,
        input_color.a
    );
}
