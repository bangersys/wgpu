//! Throwaway handoff probe for the Row-1 borrow accessor.
//!
//! NOT shippable code. It exists to answer one empirical question on a real
//! GL driver: does `Device::as_native_gl_texture` return the true GL name for
//! an own texture, and `None` where the contract says `None`?
//!
//! Run on a machine with a real GL driver (WGL default on Windows; rebuild
//! with `RUSTFLAGS="--cfg windows_angle"` for the ANGLE configuration):
//!
//! ```sh
//! cargo run -p handoff-probe
//! ```
//!
//! Paste the full stdout plus any `RUST_LOG` stderr into the review thread.

use wgpu::{Backends, InstanceDescriptor};

fn main() {
    env_logger::init();

    if std::env::var("WGPU_GL_FENCE_BEHAVIOR").is_ok() {
        println!("FAIL: WGPU_GL_FENCE_BEHAVIOR is set; AutoFinish would void completion checks. Unset it and re-run.");
        std::process::exit(3);
    }

    let instance = wgpu::Instance::new(InstanceDescriptor {
        backends: Backends::GL,
        ..InstanceDescriptor::new_without_display_handle()
    });

    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .expect("FAIL: no GL adapter (driver missing or GL < 3.3 / GLES < 3.0)");

    let info = adapter.get_info();
    println!(
        "adapter backend={:?} name={} device={}",
        info.backend, info.name, info.device
    );
    assert_eq!(
        info.backend,
        wgpu::Backend::Gl,
        "FAIL: adapter is not the GL backend"
    );

    // NOTE: `Limits::downlevel_defaults()` over-requests compute limits that
    // the GL backend reports as 0 (no compute shaders downlevel), so
    // `request_device` fails with `LimitsExceeded`. The probe creates only
    // textures, so the adapter's own limits are the correct request.
    let device_desc = wgpu::DeviceDescriptor {
        label: None,
        required_features: wgpu::Features::empty(),
        required_limits: adapter.limits(),
        default_queue: wgpu::QueueDescriptor { label: None },
        experimental_features: wgpu::ExperimentalFeatures::disabled(),
        memory_hints: wgpu::MemoryHints::MemoryUsage,
        trace: wgpu::Trace::Off,
    };
    let (device, queue) = pollster::block_on(adapter.request_device(&device_desc))
        .expect("FAIL: device request failed");
    let error_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);

    let tex_desc = |label: &'static str| wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: 64,
            height: 64,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    };

    let tex_a = device.create_texture(&tex_desc("probe-a"));
    let name_a = device
        .as_native_gl_texture(&tex_a)
        .expect("FAIL: borrow of own texture returned None");
    println!("own texture borrow: Some({})", name_a.get());

    let tex_b = device.create_texture(&tex_desc("probe-b"));
    let name_b = device
        .as_native_gl_texture(&tex_b)
        .expect("FAIL: borrow of second texture returned None");
    println!("second texture borrow: Some({})", name_b.get());
    assert_ne!(
        name_a, name_b,
        "FAIL: two textures borrowed the same GL name"
    );

    let (device_2, _q2) = pollster::block_on(adapter.request_device(&device_desc))
        .expect("FAIL: second device request failed");
    let foreign = device_2.as_native_gl_texture(&tex_a);
    println!("foreign-device borrow: {foreign:?}");
    assert!(
        foreign.is_none(),
        "FAIL: foreign-device texture did not return None (expected: same_device gate in wgpu::Device)"
    );

    let tex_rt = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("probe-rt"),
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        ..tex_desc("probe-rt")
    });
    let rt = device.as_native_gl_texture(&tex_rt);
    println!("render-attachment borrow: {rt:?} (expect None: renderbuffer, not a texture)");
    assert!(
        rt.is_none(),
        "FAIL: render-attachment borrow returned Some (pinned HAL heuristic: D2, layers == 1, color target -> renderbuffer)"
    );

    // Canary: an empty submission plus a trivial map. If this hangs, GL
    // fences never signal on this driver and no completion wait can pass;
    // if it passes but the roundtrip hangs, the workload is at fault.
    let canary = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("probe-canary"),
        size: 4,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let empty = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    queue.submit([empty.finish()]);
    let canary_slice = canary.slice(..);
    let canary_mapped = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let canary_flag = canary_mapped.clone();
    canary_slice.map_async(wgpu::MapMode::Read, move |_| {
        canary_flag.store(true, std::sync::atomic::Ordering::SeqCst);
    });
    if !wait_mapped(&device, "canary") {
        diagnose_hang();
    }
    assert!(
        canary_mapped.load(std::sync::atomic::Ordering::SeqCst),
        "FAIL: wait returned Ok but the map callback never fired (mapping bug, not borrow bug)"
    );
    println!("canary: empty submission completed");
    canary.unmap();

    // Roundtrip through the borrowed texture: `write_texture` lands bytes in
    // the GL object behind the borrowed name, `copy_texture_to_buffer` reads
    // them back. Equality proves the name denotes live storage, not just a
    // nonzero number.
    const W: u32 = 64;
    const H: u32 = 64;
    let mut pattern = vec![0u8; (W * H * 4) as usize];
    for (i, b) in pattern.iter_mut().enumerate() {
        *b = (i.wrapping_mul(7).wrapping_add(3)) as u8;
    }
    let layout = wgpu::TexelCopyBufferLayout {
        offset: 0,
        bytes_per_row: Some(W * 4),
        rows_per_image: Some(H),
    };
    let extent = wgpu::Extent3d {
        width: W,
        height: H,
        depth_or_array_layers: 1,
    };
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &tex_a,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &pattern,
        layout,
        extent,
    );
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("probe-readback"),
        size: pattern.len() as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &tex_a,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout,
        },
        extent,
    );
    queue.submit([encoder.finish()]);
    let slice = readback.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    if !wait_mapped(&device, "roundtrip") {
        diagnose_hang();
    }
    let data = slice
        .get_mapped_range()
        .expect("FAIL: buffer not mapped after completed wait (mapping bug, not borrow bug)");
    assert_eq!(data.len(), pattern.len(), "FAIL: mapped size differs");
    match data.iter().zip(pattern.iter()).position(|(a, b)| a != b) {
        None => println!("roundtrip: {} bytes identical", data.len()),
        Some(i) => {
            println!(
                "FAIL: roundtrip mismatch at byte {i}: got {}, want {} (borrow, copy, or usage bug)",
                data[i], pattern[i]
            );
            std::process::exit(4);
        }
    }
    drop(data);
    readback.unmap();

    let scope_err = pollster::block_on(error_scope.pop());
    assert!(
        scope_err.is_none(),
        "FAIL: validation error captured: {scope_err:?}"
    );
    println!("PASS: all borrow assertions held");
}

/// Bounded wait for a mapped buffer. Returns false (after printing a
/// diagnosis) instead of panicking, so a fence that never signals on a given
/// driver becomes readable evidence rather than a bare `unwrap` panic.
fn wait_mapped(device: &wgpu::Device, what: &str) -> bool {
    match device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: Some(std::time::Duration::from_secs(30)),
    }) {
        Ok(_) => true,
        Err(wgpu::PollError::Timeout) => {
            println!("HANG ({what}): submission did not complete within 30s");
            false
        }
        Err(wgpu::PollError::WrongSubmissionIndex(a, b)) => {
            println!("FAIL ({what}): wrong submission index {a} vs {b} (bookkeeping bug, not a fence hang)");
            std::process::exit(5);
        }
    }
}

fn diagnose_hang() -> ! {
    println!("The GL fence behind this wait never signaled on this driver.");
    println!("The borrow results above still stand; only GPU completion waits hang.");
    println!("Do NOT set WGPU_GL_FENCE_BEHAVIOR for verification: AutoFinish");
    println!("fires callbacks before work completes, which would void the check.");
    std::process::exit(2);
}
