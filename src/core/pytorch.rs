//! PyTorch checkpoint loading for ZIP-based format.
//!
//! This module provides native Rust loading of PyTorch `.ckpt` files
//! without requiring Python or the torch library.
//!
//! Supports PyTorch >= 1.6 checkpoints which use the ZIP-based format
//! with pickle metadata and raw tensor storage files.

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek};
use std::path::Path;

use mlx_rs::Array;
use zip::ZipArchive;

use crate::core::error::{Error, Result};

// ============================================================================
// Public API
// ============================================================================

/// Load tensors from a PyTorch checkpoint file.
///
/// Supports the modern ZIP-based format used by PyTorch >= 1.6 and NeMo.
///
/// # Arguments
/// * `path` - Path to the `.ckpt` file
///
/// # Returns
/// * HashMap mapping tensor names to mlx Arrays
///
/// # Errors
/// * If file is not a valid PyTorch ZIP checkpoint
/// * If pickle contains unsupported opcodes
///
/// # Example
///
/// ```no_run
/// # fn main() -> mlx_voice::Result<()> {
/// use mlx_voice::core::pytorch;
///
/// let weights = pytorch::load("model_weights.ckpt")?;
/// println!("Loaded {} tensors", weights.len());
/// # Ok(())
/// # }
/// ```
pub fn load(path: impl AsRef<Path>) -> Result<HashMap<String, Array>> {
    let path = path.as_ref();
    tracing::debug!("Loading PyTorch checkpoint from: {}", path.display());

    // Open ZIP archive
    let file = File::open(path)?;
    let mut archive = ZipArchive::new(file)
        .map_err(|e| Error::checkpoint(format!("Not a valid ZIP archive: {}", e)))?;

    // Find the pickle metadata file
    let pkl_path = find_pickle_file(&mut archive)?;
    tracing::debug!("Found pickle metadata at: {}", pkl_path);

    // Parse pickle to extract tensor metadata
    let pkl_data = read_zip_entry(&mut archive, &pkl_path)?;
    let tensor_infos = pickle::parse_state_dict(&pkl_data)?;
    tracing::debug!("Parsed {} tensor entries from pickle", tensor_infos.len());

    // Determine base path for storage files
    let base_path = pkl_path.trim_end_matches("data.pkl");

    // Load each tensor
    let mut tensors = HashMap::with_capacity(tensor_infos.len());

    for (name, info) in tensor_infos {
        let storage_path = format!("{}data/{}", base_path, info.storage_key);
        let storage_data = read_zip_entry(&mut archive, &storage_path)?;
        let array = load_tensor(&storage_data, &info)?;
        tensors.insert(name, array);
    }

    tracing::info!("Loaded {} tensors from checkpoint", tensors.len());
    Ok(tensors)
}

/// Check if a path is a PyTorch ZIP checkpoint.
///
/// Returns true if the file starts with ZIP magic bytes (`PK`).
pub fn is_pytorch_zip(path: impl AsRef<Path>) -> bool {
    if let Ok(mut file) = File::open(path) {
        let mut magic = [0u8; 2];
        if file.read_exact(&mut magic).is_ok() {
            return magic == [0x50, 0x4B]; // "PK"
        }
    }
    false
}

// ============================================================================
// Storage Types
// ============================================================================

/// PyTorch storage data types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageType {
    /// 32-bit floating point (torch.FloatStorage)
    Float32,
    /// 16-bit floating point (torch.HalfStorage)
    Float16,
    /// 64-bit floating point (torch.DoubleStorage)
    Float64,
    /// Brain floating point 16-bit (torch.BFloat16Storage)
    BFloat16,
    /// 64-bit signed integer (torch.LongStorage)
    Int64,
    /// 32-bit signed integer (torch.IntStorage)
    Int32,
    /// 16-bit signed integer (torch.ShortStorage)
    Int16,
    /// 8-bit signed integer (torch.CharStorage)
    Int8,
    /// 8-bit unsigned integer (torch.ByteStorage)
    UInt8,
}

impl StorageType {
    /// Parse storage type from PyTorch class name.
    pub fn from_class_name(name: &str) -> Option<Self> {
        match name {
            "FloatStorage" => Some(Self::Float32),
            "HalfStorage" => Some(Self::Float16),
            "DoubleStorage" => Some(Self::Float64),
            "BFloat16Storage" => Some(Self::BFloat16),
            "LongStorage" => Some(Self::Int64),
            "IntStorage" => Some(Self::Int32),
            "ShortStorage" => Some(Self::Int16),
            "CharStorage" => Some(Self::Int8),
            "ByteStorage" => Some(Self::UInt8),
            _ => None,
        }
    }

    /// Get the size in bytes of each element.
    pub fn byte_size(&self) -> usize {
        match self {
            Self::Float64 | Self::Int64 => 8,
            Self::Float32 | Self::Int32 => 4,
            Self::Float16 | Self::BFloat16 | Self::Int16 => 2,
            Self::Int8 | Self::UInt8 => 1,
        }
    }
}

// ============================================================================
// Tensor Metadata
// ============================================================================

/// Metadata for a tensor in the checkpoint.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    /// Storage file key (e.g., "0", "1", "2")
    pub storage_key: String,
    /// Data type
    pub storage_type: StorageType,
    /// Tensor shape
    pub shape: Vec<i64>,
    /// Tensor strides
    pub stride: Vec<i64>,
    /// Number of elements in the underlying storage (which may be shared by
    /// several tensors)
    pub numel: usize,
    /// Offset into the storage, in elements (usually 0)
    pub offset: usize,
}

// ============================================================================
// Internal Helpers
// ============================================================================

/// Find the data.pkl file in the archive.
fn find_pickle_file<R: Read + Seek>(archive: &mut ZipArchive<R>) -> Result<String> {
    for name in archive.file_names().flatten() {
        if name.ends_with("data.pkl") {
            return Ok(name.into_owned());
        }
    }
    Err(Error::checkpoint("No data.pkl found in checkpoint archive"))
}

/// Read a ZIP entry to bytes.
fn read_zip_entry<R: Read + Seek>(archive: &mut ZipArchive<R>, name: &str) -> Result<Vec<u8>> {
    let mut entry = archive
        .by_name(name)
        .map_err(|e| Error::checkpoint(format!("Failed to read '{}': {}", name, e)))?;

    let mut data = Vec::with_capacity(entry.size() as usize);
    entry.read_to_end(&mut data)?;
    Ok(data)
}

/// Extract a tensor's bytes, in row-major order, from its storage.
///
/// Tensors can be views into a shared storage (e.g. cuDNN-flattened LSTM
/// weights), so this honors the storage offset and strides.
fn gather_tensor_bytes(data: &[u8], info: &TensorInfo, elem_size: usize) -> Result<Vec<u8>> {
    let numel: usize = info.shape.iter().map(|&d| d.max(0) as usize).product();
    if numel == 0 {
        return Ok(Vec::new());
    }

    // Row-major strides for this shape
    let mut contiguous = vec![1i64; info.shape.len()];
    for i in (0..info.shape.len().saturating_sub(1)).rev() {
        contiguous[i] = contiguous[i + 1] * info.shape[i + 1];
    }
    let stride = if info.stride.len() == info.shape.len() {
        info.stride.clone()
    } else {
        contiguous.clone()
    };

    // Last element touched, to bounds-check the storage
    let max_index = info.offset
        + info
            .shape
            .iter()
            .zip(&stride)
            .map(|(&d, &s)| (d.max(1) - 1) as usize * s.max(0) as usize)
            .sum::<usize>();
    if data.len() < (max_index + 1) * elem_size {
        return Err(Error::checkpoint(format!(
            "Storage too small: expected {} bytes, got {}",
            (max_index + 1) * elem_size,
            data.len()
        )));
    }

    let is_contiguous = info
        .shape
        .iter()
        .zip(stride.iter().zip(&contiguous))
        .all(|(&d, (&s, &c))| d == 1 || s == c);
    if is_contiguous {
        let start = info.offset * elem_size;
        return Ok(data[start..start + numel * elem_size].to_vec());
    }

    let mut out = Vec::with_capacity(numel * elem_size);
    let mut index = vec![0i64; info.shape.len()];
    for _ in 0..numel {
        let elem =
            info.offset as i64 + index.iter().zip(&stride).map(|(&i, &s)| i * s).sum::<i64>();
        let start = elem as usize * elem_size;
        out.extend_from_slice(&data[start..start + elem_size]);
        // Advance the multi-dimensional index
        for dim in (0..index.len()).rev() {
            index[dim] += 1;
            if index[dim] < info.shape[dim] {
                break;
            }
            index[dim] = 0;
        }
    }
    Ok(out)
}

/// Load a tensor from raw storage bytes.
fn load_tensor(data: &[u8], info: &TensorInfo) -> Result<Array> {
    let elem_size = info.storage_type.byte_size();
    let gathered = gather_tensor_bytes(data, info, elem_size)?;
    let storage_bytes: &[u8] = &gathered;

    // Convert shape to i32 for mlx-rs
    let shape: Vec<i32> = info.shape.iter().map(|&d| d as i32).collect();

    // Create array based on dtype
    let array = match info.storage_type {
        StorageType::Float32 => {
            let floats: &[f32] = bytemuck::cast_slice(storage_bytes);
            Array::from_slice(floats, &shape)
        }
        StorageType::Float64 => {
            // Convert f64 to f32 for mlx
            let doubles: &[f64] = bytemuck::cast_slice(storage_bytes);
            let floats: Vec<f32> = doubles.iter().map(|&x| x as f32).collect();
            Array::from_slice(&floats, &shape)
        }
        StorageType::Int64 => {
            let ints: &[i64] = bytemuck::cast_slice(storage_bytes);
            // MLX uses i32, convert
            let ints32: Vec<i32> = ints.iter().map(|&x| x as i32).collect();
            Array::from_slice(&ints32, &shape)
        }
        StorageType::Int32 => {
            let ints: &[i32] = bytemuck::cast_slice(storage_bytes);
            Array::from_slice(ints, &shape)
        }
        StorageType::Float16 => {
            // For f16, we need to convert to f32
            // f16 is stored as u16 bit pattern
            let raw: &[u16] = bytemuck::cast_slice(storage_bytes);
            let floats: Vec<f32> = raw
                .iter()
                .map(|&bits| half::f16::from_bits(bits).to_f32())
                .collect();
            Array::from_slice(&floats, &shape)
        }
        StorageType::BFloat16 => {
            // For bf16, convert to f32
            let raw: &[u16] = bytemuck::cast_slice(storage_bytes);
            let floats: Vec<f32> = raw
                .iter()
                .map(|&bits| half::bf16::from_bits(bits).to_f32())
                .collect();
            Array::from_slice(&floats, &shape)
        }
        StorageType::Int16 => {
            let ints: &[i16] = bytemuck::cast_slice(storage_bytes);
            let ints32: Vec<i32> = ints.iter().map(|&x| x as i32).collect();
            Array::from_slice(&ints32, &shape)
        }
        StorageType::Int8 => {
            let ints: &[i8] = bytemuck::cast_slice(storage_bytes);
            let ints32: Vec<i32> = ints.iter().map(|&x| x as i32).collect();
            Array::from_slice(&ints32, &shape)
        }
        StorageType::UInt8 => {
            let ints: &[u8] = storage_bytes;
            let ints32: Vec<i32> = ints.iter().map(|&x| x as i32).collect();
            Array::from_slice(&ints32, &shape)
        }
    };

    Ok(array)
}

// ============================================================================
// Pickle Parser Module
// ============================================================================

mod pickle {
    //! Minimal pickle parser for PyTorch state_dict files.
    //!
    //! This implements only the subset of pickle protocol 2 opcodes needed
    //! to parse PyTorch checkpoint metadata.

    use super::*;

    // Pickle opcodes we need to handle
    const PROTO: u8 = 0x80;
    const FRAME: u8 = 0x95;
    const GLOBAL: u8 = b'c';
    const REDUCE: u8 = b'R';
    const BINPUT: u8 = b'q';
    const BINGET: u8 = b'h';
    const LONG_BINPUT: u8 = b'r';
    const LONG_BINGET: u8 = b'j';
    const SHORT_BINUNICODE: u8 = 0x8c;
    const BINUNICODE: u8 = b'X';
    const BININT1: u8 = b'K';
    const BININT2: u8 = b'M';
    const BININT: u8 = b'J';
    const BINPERSID: u8 = b'Q';
    const TUPLE: u8 = b't';
    const TUPLE1: u8 = 0x85;
    const TUPLE2: u8 = 0x86;
    const TUPLE3: u8 = 0x87;
    const EMPTY_TUPLE: u8 = b')';
    const EMPTY_LIST: u8 = b']';
    const EMPTY_DICT: u8 = b'}';
    const MARK: u8 = b'(';
    const SETITEM: u8 = b's';
    const SETITEMS: u8 = b'u';
    const APPEND: u8 = b'a';
    const APPENDS: u8 = b'e';
    const NEWFALSE: u8 = 0x89;
    const NEWTRUE: u8 = 0x88;
    const NONE: u8 = b'N';
    const STOP: u8 = b'.';
    const MEMOIZE: u8 = 0x94;
    const BUILD: u8 = b'b';
    const BINFLOAT: u8 = b'G';

    /// Values that can appear on the pickle stack
    #[derive(Debug, Clone)]
    #[allow(dead_code)] // Some variants are parsed but values not used
    pub enum Value {
        None,
        Bool(bool),
        Int(i64),
        Float(f64),
        String(String),
        Tuple(Vec<Value>),
        List(Vec<Value>),
        Dict(Vec<(String, Value)>),
        Mark,
        Global { module: String, name: String },
        StorageRef(StorageRef),
        TensorMeta(TensorInfo),
        Ignored,
    }

    /// Reference to a storage location
    #[derive(Debug, Clone)]
    pub struct StorageRef {
        pub storage_type: StorageType,
        pub key: String,
        pub numel: usize,
    }

    /// Parse a PyTorch state_dict pickle into tensor metadata.
    pub fn parse_state_dict(data: &[u8]) -> Result<HashMap<String, TensorInfo>> {
        let mut parser = Parser::new(data);
        parser.parse()
    }

    struct Parser<'a> {
        data: &'a [u8],
        pos: usize,
        stack: Vec<Value>,
        memo: HashMap<u32, Value>,
    }

    impl<'a> Parser<'a> {
        fn new(data: &'a [u8]) -> Self {
            Self {
                data,
                pos: 0,
                stack: Vec::with_capacity(64),
                memo: HashMap::new(),
            }
        }

        fn parse(&mut self) -> Result<HashMap<String, TensorInfo>> {
            while self.pos < self.data.len() {
                let opcode = self.read_u8()?;

                match opcode {
                    PROTO => {
                        let _version = self.read_u8()?;
                    }
                    FRAME => {
                        // Skip frame size (8 bytes)
                        self.pos += 8;
                    }
                    GLOBAL => self.handle_global()?,
                    REDUCE => self.handle_reduce()?,
                    BUILD => self.handle_build()?,
                    BINPUT => {
                        let key = self.read_u8()? as u32;
                        if let Some(val) = self.stack.last().cloned() {
                            self.memo.insert(key, val);
                        }
                    }
                    LONG_BINPUT => {
                        let key = self.read_u32_le()?;
                        if let Some(val) = self.stack.last().cloned() {
                            self.memo.insert(key, val);
                        }
                    }
                    BINGET => {
                        let key = self.read_u8()? as u32;
                        let val = self.memo.get(&key).cloned().unwrap_or(Value::None);
                        self.stack.push(val);
                    }
                    LONG_BINGET => {
                        let key = self.read_u32_le()?;
                        let val = self.memo.get(&key).cloned().unwrap_or(Value::None);
                        self.stack.push(val);
                    }
                    MEMOIZE => {
                        let key = self.memo.len() as u32;
                        if let Some(val) = self.stack.last().cloned() {
                            self.memo.insert(key, val);
                        }
                    }
                    SHORT_BINUNICODE => {
                        let len = self.read_u8()? as usize;
                        let s = self.read_string(len)?;
                        self.stack.push(Value::String(s));
                    }
                    BINUNICODE => {
                        let len = self.read_u32_le()? as usize;
                        let s = self.read_string(len)?;
                        self.stack.push(Value::String(s));
                    }
                    BININT1 => {
                        let val = self.read_u8()? as i64;
                        self.stack.push(Value::Int(val));
                    }
                    BININT2 => {
                        let val = self.read_u16_le()? as i64;
                        self.stack.push(Value::Int(val));
                    }
                    BININT => {
                        let val = self.read_i32_le()? as i64;
                        self.stack.push(Value::Int(val));
                    }
                    BINFLOAT => {
                        let val = self.read_f64_be()?;
                        self.stack.push(Value::Float(val));
                    }
                    BINPERSID => self.handle_binpersid()?,
                    TUPLE => self.handle_tuple()?,
                    TUPLE1 => self.handle_tuple_n(1)?,
                    TUPLE2 => self.handle_tuple_n(2)?,
                    TUPLE3 => self.handle_tuple_n(3)?,
                    EMPTY_TUPLE => {
                        self.stack.push(Value::Tuple(Vec::new()));
                    }
                    EMPTY_LIST => {
                        self.stack.push(Value::List(Vec::new()));
                    }
                    EMPTY_DICT => {
                        self.stack.push(Value::Dict(Vec::new()));
                    }
                    MARK => {
                        self.stack.push(Value::Mark);
                    }
                    SETITEM => self.handle_setitem()?,
                    SETITEMS => self.handle_setitems()?,
                    APPEND => self.handle_append()?,
                    APPENDS => self.handle_appends()?,
                    NEWFALSE => {
                        self.stack.push(Value::Bool(false));
                    }
                    NEWTRUE => {
                        self.stack.push(Value::Bool(true));
                    }
                    NONE => {
                        self.stack.push(Value::None);
                    }
                    STOP => break,
                    _ => {
                        return Err(Error::pickle(format!(
                            "Unsupported pickle opcode: 0x{:02x} at position {}",
                            opcode,
                            self.pos - 1
                        )));
                    }
                }
            }

            // Extract tensor metadata from the final state dict
            self.extract_tensors()
        }

        fn handle_global(&mut self) -> Result<()> {
            let module = self.read_line()?;
            let name = self.read_line()?;
            self.stack.push(Value::Global {
                module: module.to_string(),
                name: name.to_string(),
            });
            Ok(())
        }

        fn handle_reduce(&mut self) -> Result<()> {
            let args = self
                .stack
                .pop()
                .ok_or_else(|| Error::pickle("Stack underflow in REDUCE"))?;
            let callable = self
                .stack
                .pop()
                .ok_or_else(|| Error::pickle("Stack underflow in REDUCE"))?;

            match callable {
                Value::Global { module, name } => {
                    if module == "torch._utils" && name == "_rebuild_tensor_v2" {
                        // Parse tensor reconstruction args
                        if let Value::Tuple(args) = args {
                            let tensor = self.parse_tensor_args(&args)?;
                            self.stack.push(Value::TensorMeta(tensor));
                        } else {
                            self.stack.push(Value::Ignored);
                        }
                    } else if module == "collections" && name == "OrderedDict" {
                        // Create empty dict for OrderedDict()
                        self.stack.push(Value::Dict(Vec::new()));
                    } else {
                        // Unknown callable, push ignored value
                        self.stack.push(Value::Ignored);
                    }
                }
                _ => {
                    self.stack.push(Value::Ignored);
                }
            }
            Ok(())
        }

        fn handle_build(&mut self) -> Result<()> {
            // BUILD is used to set state on an object
            // For our purposes, we can ignore the state argument
            let _state = self.stack.pop();
            // The object being built remains on stack
            Ok(())
        }

        fn handle_binpersid(&mut self) -> Result<()> {
            // Persistent ID - the top of stack is the ID tuple
            let pid = self
                .stack
                .pop()
                .ok_or_else(|| Error::pickle("Stack underflow in BINPERSID"))?;

            if let Value::Tuple(items) = pid {
                // Expected format: ('storage', StorageType, key, device, numel)
                if items.len() >= 5 {
                    if let (Value::Global { name, .. }, Value::String(key), Value::Int(numel)) =
                        (&items[1], &items[2], &items[4])
                    {
                        if let Some(storage_type) = StorageType::from_class_name(name) {
                            self.stack.push(Value::StorageRef(StorageRef {
                                storage_type,
                                key: key.clone(),
                                numel: *numel as usize,
                            }));
                            return Ok(());
                        }
                    }
                }
            }

            // If we can't parse it, push ignored
            self.stack.push(Value::Ignored);
            Ok(())
        }

        fn handle_tuple(&mut self) -> Result<()> {
            // Find mark and collect items
            let mut items = Vec::new();
            while let Some(val) = self.stack.pop() {
                if matches!(val, Value::Mark) {
                    break;
                }
                items.push(val);
            }
            items.reverse();
            self.stack.push(Value::Tuple(items));
            Ok(())
        }

        fn handle_tuple_n(&mut self, n: usize) -> Result<()> {
            let mut items = Vec::with_capacity(n);
            for _ in 0..n {
                let val = self
                    .stack
                    .pop()
                    .ok_or_else(|| Error::pickle("Stack underflow in TUPLE_N"))?;
                items.push(val);
            }
            items.reverse();
            self.stack.push(Value::Tuple(items));
            Ok(())
        }

        fn handle_setitem(&mut self) -> Result<()> {
            let value = self
                .stack
                .pop()
                .ok_or_else(|| Error::pickle("Stack underflow in SETITEM"))?;
            let key = self
                .stack
                .pop()
                .ok_or_else(|| Error::pickle("Stack underflow in SETITEM"))?;

            if let Some(Value::Dict(dict)) = self.stack.last_mut() {
                if let Value::String(k) = key {
                    dict.push((k, value));
                }
            }
            Ok(())
        }

        fn handle_setitems(&mut self) -> Result<()> {
            // Collect key-value pairs until mark
            let mut pairs = Vec::new();
            loop {
                let val = self
                    .stack
                    .pop()
                    .ok_or_else(|| Error::pickle("Stack underflow in SETITEMS"))?;
                if matches!(val, Value::Mark) {
                    break;
                }
                let key = self
                    .stack
                    .pop()
                    .ok_or_else(|| Error::pickle("Stack underflow in SETITEMS"))?;
                if let Value::String(k) = key {
                    pairs.push((k, val));
                }
            }

            // The dict should be on top now
            if let Some(Value::Dict(dict)) = self.stack.last_mut() {
                pairs.reverse();
                dict.extend(pairs);
            }
            Ok(())
        }

        fn handle_append(&mut self) -> Result<()> {
            let value = self
                .stack
                .pop()
                .ok_or_else(|| Error::pickle("Stack underflow in APPEND"))?;
            if let Some(Value::List(list)) = self.stack.last_mut() {
                list.push(value);
            }
            Ok(())
        }

        fn handle_appends(&mut self) -> Result<()> {
            let mut items = Vec::new();
            loop {
                let val = self
                    .stack
                    .pop()
                    .ok_or_else(|| Error::pickle("Stack underflow in APPENDS"))?;
                if matches!(val, Value::Mark) {
                    break;
                }
                items.push(val);
            }
            items.reverse();

            if let Some(Value::List(list)) = self.stack.last_mut() {
                list.extend(items);
            }
            Ok(())
        }

        fn parse_tensor_args(&self, args: &[Value]) -> Result<TensorInfo> {
            // Expected: (storage_ref, offset, shape, stride, requires_grad, backward_hooks)
            if args.len() < 4 {
                return Err(Error::pickle("Invalid tensor args: too few elements"));
            }

            let storage = match &args[0] {
                Value::StorageRef(s) => s.clone(),
                Value::Tuple(t) if t.len() >= 3 => {
                    // Sometimes storage is nested in a tuple
                    if let (Value::Global { name, .. }, Value::String(key), Value::Int(numel)) =
                        (&t[1], &t[2], &t[4])
                    {
                        StorageRef {
                            storage_type: StorageType::from_class_name(name).ok_or_else(|| {
                                Error::pickle(format!("Unknown storage type: {}", name))
                            })?,
                            key: key.clone(),
                            numel: *numel as usize,
                        }
                    } else {
                        return Err(Error::pickle("Invalid storage tuple format"));
                    }
                }
                _ => return Err(Error::pickle("Expected storage ref as first tensor arg")),
            };

            let offset = match &args[1] {
                Value::Int(n) => *n as usize,
                _ => 0,
            };

            let shape = match &args[2] {
                Value::Tuple(dims) => dims
                    .iter()
                    .filter_map(|v| match v {
                        Value::Int(n) => Some(*n),
                        _ => None,
                    })
                    .collect(),
                _ => return Err(Error::pickle("Expected tuple for tensor shape")),
            };

            let stride = match &args[3] {
                Value::Tuple(dims) => dims
                    .iter()
                    .filter_map(|v| match v {
                        Value::Int(n) => Some(*n),
                        _ => None,
                    })
                    .collect(),
                _ => Vec::new(),
            };

            Ok(TensorInfo {
                storage_key: storage.key,
                storage_type: storage.storage_type,
                shape,
                stride,
                numel: storage.numel,
                offset,
            })
        }

        fn extract_tensors(&self) -> Result<HashMap<String, TensorInfo>> {
            let mut tensors = HashMap::new();

            // The final value on the stack should be the state dict
            if let Some(Value::Dict(pairs)) = self.stack.last() {
                for (name, value) in pairs {
                    if let Value::TensorMeta(info) = value {
                        tensors.insert(name.clone(), info.clone());
                    }
                }
            }

            if tensors.is_empty() {
                return Err(Error::pickle("No tensors found in pickle"));
            }

            Ok(tensors)
        }

        // Byte reading helpers

        fn read_u8(&mut self) -> Result<u8> {
            if self.pos >= self.data.len() {
                return Err(Error::pickle("Unexpected end of pickle data"));
            }
            let val = self.data[self.pos];
            self.pos += 1;
            Ok(val)
        }

        fn read_u16_le(&mut self) -> Result<u16> {
            if self.pos + 2 > self.data.len() {
                return Err(Error::pickle("Unexpected end of pickle data"));
            }
            let val = u16::from_le_bytes([self.data[self.pos], self.data[self.pos + 1]]);
            self.pos += 2;
            Ok(val)
        }

        fn read_u32_le(&mut self) -> Result<u32> {
            if self.pos + 4 > self.data.len() {
                return Err(Error::pickle("Unexpected end of pickle data"));
            }
            let val = u32::from_le_bytes([
                self.data[self.pos],
                self.data[self.pos + 1],
                self.data[self.pos + 2],
                self.data[self.pos + 3],
            ]);
            self.pos += 4;
            Ok(val)
        }

        fn read_i32_le(&mut self) -> Result<i32> {
            if self.pos + 4 > self.data.len() {
                return Err(Error::pickle("Unexpected end of pickle data"));
            }
            let val = i32::from_le_bytes([
                self.data[self.pos],
                self.data[self.pos + 1],
                self.data[self.pos + 2],
                self.data[self.pos + 3],
            ]);
            self.pos += 4;
            Ok(val)
        }

        fn read_f64_be(&mut self) -> Result<f64> {
            if self.pos + 8 > self.data.len() {
                return Err(Error::pickle("Unexpected end of pickle data"));
            }
            let val = f64::from_be_bytes([
                self.data[self.pos],
                self.data[self.pos + 1],
                self.data[self.pos + 2],
                self.data[self.pos + 3],
                self.data[self.pos + 4],
                self.data[self.pos + 5],
                self.data[self.pos + 6],
                self.data[self.pos + 7],
            ]);
            self.pos += 8;
            Ok(val)
        }

        fn read_string(&mut self, len: usize) -> Result<String> {
            if self.pos + len > self.data.len() {
                return Err(Error::pickle("Unexpected end of pickle data"));
            }
            let bytes = &self.data[self.pos..self.pos + len];
            let s = std::str::from_utf8(bytes)
                .map_err(|e| Error::pickle(format!("Invalid UTF-8 in string: {}", e)))?;
            self.pos += len;
            Ok(s.to_string())
        }

        fn read_line(&mut self) -> Result<&'a str> {
            let start = self.pos;
            while self.pos < self.data.len() && self.data[self.pos] != b'\n' {
                self.pos += 1;
            }
            let bytes = &self.data[start..self.pos];
            self.pos += 1; // Skip newline
            std::str::from_utf8(bytes)
                .map_err(|e| Error::pickle(format!("Invalid UTF-8 in line: {}", e)))
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn info(shape: Vec<i64>, stride: Vec<i64>, offset: usize) -> TensorInfo {
        TensorInfo {
            storage_key: "0".to_string(),
            storage_type: StorageType::Float32,
            shape,
            stride,
            numel: 12,
            offset,
        }
    }

    fn storage() -> Vec<u8> {
        (0..12).flat_map(|i| (i as f32).to_le_bytes()).collect()
    }

    fn values(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn test_gather_view_into_shared_storage() {
        // A [2, 2] tensor starting at element 4 of a 12-element storage
        let bytes = gather_tensor_bytes(&storage(), &info(vec![2, 2], vec![2, 1], 4), 4).unwrap();
        assert_eq!(values(&bytes), vec![4.0, 5.0, 6.0, 7.0]);
    }

    #[test]
    fn test_gather_transposed_view() {
        // Transpose of a [2, 3] tensor: shape [3, 2], strides [1, 3]
        let bytes = gather_tensor_bytes(&storage(), &info(vec![3, 2], vec![1, 3], 0), 4).unwrap();
        assert_eq!(values(&bytes), vec![0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);
    }

    #[test]
    fn test_gather_rejects_short_storage() {
        assert!(gather_tensor_bytes(&storage(), &info(vec![4], vec![1], 10), 4).is_err());
    }

    #[test]
    fn test_storage_type_from_name() {
        assert_eq!(
            StorageType::from_class_name("FloatStorage"),
            Some(StorageType::Float32)
        );
        assert_eq!(
            StorageType::from_class_name("LongStorage"),
            Some(StorageType::Int64)
        );
        assert_eq!(
            StorageType::from_class_name("HalfStorage"),
            Some(StorageType::Float16)
        );
        assert_eq!(
            StorageType::from_class_name("BFloat16Storage"),
            Some(StorageType::BFloat16)
        );
        assert_eq!(StorageType::from_class_name("Unknown"), None);
    }

    #[test]
    fn test_storage_byte_size() {
        assert_eq!(StorageType::Float32.byte_size(), 4);
        assert_eq!(StorageType::Float64.byte_size(), 8);
        assert_eq!(StorageType::Int64.byte_size(), 8);
        assert_eq!(StorageType::Float16.byte_size(), 2);
        assert_eq!(StorageType::BFloat16.byte_size(), 2);
        assert_eq!(StorageType::Int8.byte_size(), 1);
    }

    #[test]
    fn test_is_pytorch_zip() {
        // Non-existent file should return false
        assert!(!is_pytorch_zip("/nonexistent/path"));
    }
}
