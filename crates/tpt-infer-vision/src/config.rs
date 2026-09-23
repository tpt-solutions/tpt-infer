//! Preprocessing configuration: target size, normalization, layout, presets.

/// Output memory layout of the preprocessed tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Layout {
    /// Channels-first `[C, H, W]` (batched `[1, C, H, W]`).
    #[default]
    Chw,
    /// Channels-last `[H, W, C]` (batched `[1, H, W, C]`).
    Hwc,
}

/// Channel order of the output pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ColorOrder {
    /// Red, green, blue (decoder order).
    #[default]
    Rgb,
    /// Blue, green, red (OpenCV/Caffe style).
    Bgr,
}

/// Configuration for [`preprocess_image`](crate::preprocess_image) and
/// [`preprocess_raw_rgb`](crate::preprocess_raw_rgb).
///
/// Pipeline order: (optional letterbox) → rescale → per-channel
/// `(x - mean) / std` → layout conversion.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PreprocessConfig {
    /// Target width in pixels.
    pub width: u32,
    /// Target height in pixels.
    pub height: u32,
    /// Per-channel mean, expressed in the order given by [`ColorOrder`]
    /// *after* applying [`rescale`](Self::rescale).
    pub mean: [f32; 3],
    /// Per-channel standard deviation; every entry must be non-zero.
    pub std: [f32; 3],
    /// Output layout.
    pub layout: Layout,
    /// Letterbox (aspect-preserving resize + pad) instead of a plain stretch.
    pub letterbox: bool,
    /// Multiplier applied to pixel values before normalization
    /// (e.g. `1.0 / 255.0` to map `[0, 255]` into `[0, 1]`).
    pub rescale: f32,
    /// Output channel order.
    pub color: ColorOrder,
    /// Letterbox fill value in input pixel units (pre-rescale),
    /// e.g. `114.0` for the YOLO convention. Ignored when
    /// [`letterbox`](Self::letterbox) is `false`.
    pub pad: f32,
}

impl PreprocessConfig {
    /// ImageNet / torchvision defaults: 224×224 stretch resize, CHW RGB,
    /// `1/255` rescale, mean/std in `[0, 1]` space, no letterbox.
    pub fn imagenet() -> Self {
        Self {
            width: 224,
            height: 224,
            mean: [0.485, 0.456, 0.406],
            std: [0.229, 0.224, 0.225],
            layout: Layout::Chw,
            letterbox: false,
            rescale: 1.0 / 255.0,
            color: ColorOrder::Rgb,
            pad: 0.0,
        }
    }

    /// YOLO-style defaults: 640×640 letterbox with pad `114`, `1/255`
    /// rescale, identity normalization, CHW RGB.
    pub fn yolo() -> Self {
        Self {
            width: 640,
            height: 640,
            mean: [0.0, 0.0, 0.0],
            std: [1.0, 1.0, 1.0],
            layout: Layout::Chw,
            letterbox: true,
            rescale: 1.0 / 255.0,
            color: ColorOrder::Rgb,
            pad: 114.0,
        }
    }
}

impl Default for PreprocessConfig {
    fn default() -> Self {
        Self::imagenet()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn imagenet_preset() {
        let c = PreprocessConfig::imagenet();
        assert_eq!(c.width, 224);
        assert_eq!(c.height, 224);
        assert_eq!(c.mean, [0.485, 0.456, 0.406]);
        assert_eq!(c.std, [0.229, 0.224, 0.225]);
        assert_eq!(c.layout, Layout::Chw);
        assert!(!c.letterbox);
        assert!((c.rescale - 1.0 / 255.0).abs() < f32::EPSILON);
        assert_eq!(c.color, ColorOrder::Rgb);
        assert_eq!(c.pad, 0.0);
    }

    #[test]
    fn yolo_preset() {
        let c = PreprocessConfig::yolo();
        assert_eq!(c.width, 640);
        assert_eq!(c.height, 640);
        assert_eq!(c.mean, [0.0, 0.0, 0.0]);
        assert_eq!(c.std, [1.0, 1.0, 1.0]);
        assert_eq!(c.layout, Layout::Chw);
        assert!(c.letterbox);
        assert!((c.rescale - 1.0 / 255.0).abs() < f32::EPSILON);
        assert_eq!(c.color, ColorOrder::Rgb);
        assert_eq!(c.pad, 114.0);
    }

    #[test]
    fn default_is_imagenet() {
        assert_eq!(PreprocessConfig::default(), PreprocessConfig::imagenet());
    }

    #[test]
    fn layout_and_color_defaults() {
        assert_eq!(Layout::default(), Layout::Chw);
        assert_eq!(ColorOrder::default(), ColorOrder::Rgb);
    }
}
