//! Element data types supported by tpt-infer tensors.

/// Data type of tensor elements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DType {
    /// 32-bit IEEE 754 floating point.
    F32,
    /// 16-bit IEEE 754 floating point (stored as `u16` bits).
    F16,
    /// 8-bit signed integer.
    I8,
    /// 4-bit signed integer (packed two-per-byte).
    I4,
    /// 32-bit signed integer.
    I32,
    /// 64-bit signed integer.
    I64,
    /// Unsigned 8-bit integer.
    U8,
    /// Boolean (stored as `u8` 0/1).
    Bool,
}

impl DType {
    /// Size of one element in bits.
    pub const fn size_bits(self) -> usize {
        match self {
            DType::F32 | DType::I32 => 32,
            DType::F16 => 16,
            DType::I8 | DType::U8 | DType::Bool => 8,
            DType::I4 => 4,
            DType::I64 => 64,
        }
    }

    /// Size of one element in bytes (rounded up; `I4` reports 1 for a whole byte).
    pub const fn size_bytes(self) -> usize {
        self.size_bits().div_ceil(8)
    }

    /// Canonical lowercase name (e.g. `"f32"`).
    pub const fn name(self) -> &'static str {
        match self {
            DType::F32 => "f32",
            DType::F16 => "f16",
            DType::I8 => "i8",
            DType::I4 => "i4",
            DType::I32 => "i32",
            DType::I64 => "i64",
            DType::U8 => "u8",
            DType::Bool => "bool",
        }
    }

    /// Parse from a canonical name.
    pub fn from_name(s: &str) -> Option<Self> {
        match s {
            "f32" => Some(DType::F32),
            "f16" => Some(DType::F16),
            "i8" => Some(DType::I8),
            "i4" => Some(DType::I4),
            "i32" => Some(DType::I32),
            "i64" => Some(DType::I64),
            "u8" => Some(DType::U8),
            "bool" => Some(DType::Bool),
            _ => None,
        }
    }

    /// Whether this is a floating-point type.
    pub const fn is_float(self) -> bool {
        matches!(self, DType::F32 | DType::F16)
    }

    /// Whether this is an integer type.
    pub const fn is_int(self) -> bool {
        matches!(self, DType::I8 | DType::I4 | DType::I32 | DType::I64 | DType::U8)
    }
}

impl core::fmt::Display for DType {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtype_sizes() {
        assert_eq!(DType::F32.size_bytes(), 4);
        assert_eq!(DType::I4.size_bits(), 4);
        assert_eq!(DType::I4.size_bytes(), 1);
        assert_eq!(DType::F16.size_bytes(), 2);
    }

    #[test]
    fn dtype_roundtrip_name() {
        for dt in [
            DType::F32,
            DType::F16,
            DType::I8,
            DType::I4,
            DType::I32,
            DType::I64,
            DType::U8,
            DType::Bool,
        ] {
            assert_eq!(DType::from_name(dt.name()), Some(dt));
        }
        assert_eq!(DType::from_name("nope"), None);
    }
}
