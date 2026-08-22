import itertools
import os
import re
import torch
import json
import struct
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Union, Optional

from safetensors.torch import load_file


def mem_eff_save_file(tensors: Dict[str, torch.Tensor], filename: str, metadata: Dict[str, Any] = None):
    """
    memory efficient save file
    """

    _TYPES = {
        torch.float64: "F64",
        torch.float32: "F32",
        torch.float16: "F16",
        torch.bfloat16: "BF16",
        torch.int64: "I64",
        torch.int32: "I32",
        torch.int16: "I16",
        torch.int8: "I8",
        torch.uint8: "U8",
        torch.bool: "BOOL",
        getattr(torch, "float8_e5m2", None): "F8_E5M2",
        getattr(torch, "float8_e4m3fn", None): "F8_E4M3",
    }
    _ALIGN = 256

    def validate_metadata(metadata: Dict[str, Any]) -> Dict[str, str]:
        validated = {}
        for key, value in metadata.items():
            if not isinstance(key, str):
                raise ValueError(f"Metadata key must be a string, got {type(key)}")
            if not isinstance(value, str):
                print(f"Warning: Metadata value for key '{key}' is not a string. Converting to string.")
                validated[key] = str(value)
            else:
                validated[key] = value
        return validated

    # print(f"Using memory efficient save file: {filename}")

    header = {}
    offset = 0
    if metadata:
        header["__metadata__"] = validate_metadata(metadata)
    for k, v in tensors.items():
        if v.numel() == 0:  # empty tensor
            header[k] = {"dtype": _TYPES[v.dtype], "shape": list(v.shape), "data_offsets": [offset, offset]}
        else:
            size = v.numel() * v.element_size()
            header[k] = {"dtype": _TYPES[v.dtype], "shape": list(v.shape), "data_offsets": [offset, offset + size]}
            offset += size

    hjson = json.dumps(header).encode("utf-8")
    hjson += b" " * (-(len(hjson) + 8) % _ALIGN)

    with open(filename, "wb") as f:
        f.write(struct.pack("<Q", len(hjson)))
        f.write(hjson)

        for k, v in tensors.items():
            if v.numel() == 0:
                continue
            if v.is_cuda:
                # Direct GPU to disk save
                with torch.cuda.device(v.device):
                    if v.dim() == 0:  # if scalar, need to add a dimension to work with view
                        v = v.unsqueeze(0)
                    tensor_bytes = v.contiguous().view(torch.uint8)
                    tensor_bytes.cpu().numpy().tofile(f)
            else:
                # CPU tensor save
                if v.dim() == 0:  # if scalar, need to add a dimension to work with view
                    v = v.unsqueeze(0)
                v.contiguous().view(torch.uint8).numpy().tofile(f)


class MemoryEfficientSafeOpen:
    # does not support metadata loading
    def __init__(self, filename):
        self.filename = filename
        self.file = open(filename, "rb")
        self.header, self.header_size = self._read_header()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()

    def keys(self):
        return [k for k in self.header.keys() if k != "__metadata__"]

    def metadata(self) -> Dict[str, str]:
        return self.header.get("__metadata__", {})

    def get_tensor(self, key):
        if key not in self.header:
            raise KeyError(f"Tensor '{key}' not found in the file")

        metadata = self.header[key]
        offset_start, offset_end = metadata["data_offsets"]

        if offset_start == offset_end:
            tensor_bytes = None
        else:
            # adjust offset by header size
            self.file.seek(self.header_size + 8 + offset_start)
            tensor_bytes = bytearray(offset_end - offset_start)
            n = self.file.readinto(tensor_bytes)
            if n != len(tensor_bytes):
                raise IOError(f"short read for tensor '{key}' in {self.filename}")

        return self._deserialize_tensor(tensor_bytes, metadata)

    def _read_header(self):
        header_size = struct.unpack("<Q", self.file.read(8))[0]
        header_json = self.file.read(header_size).decode("utf-8")
        return json.loads(header_json), header_size

    def _deserialize_tensor(self, tensor_bytes, metadata):
        dtype = self._get_torch_dtype(metadata["dtype"])
        shape = metadata["shape"]

        if tensor_bytes is None:
            byte_tensor = torch.empty(0, dtype=torch.uint8)
        else:
            if not isinstance(tensor_bytes, bytearray):
                tensor_bytes = bytearray(tensor_bytes)  # make it writable
            byte_tensor = torch.frombuffer(tensor_bytes, dtype=torch.uint8)

        # process float8 types
        if metadata["dtype"] in ["F8_E5M2", "F8_E4M3"]:
            return self._convert_float8(byte_tensor, metadata["dtype"], shape)

        # convert to the target dtype and reshape
        return byte_tensor.view(dtype).reshape(shape)

    @staticmethod
    def _get_torch_dtype(dtype_str):
        dtype_map = {
            "F64": torch.float64,
            "F32": torch.float32,
            "F16": torch.float16,
            "BF16": torch.bfloat16,
            "I64": torch.int64,
            "I32": torch.int32,
            "I16": torch.int16,
            "I8": torch.int8,
            "U8": torch.uint8,
            "BOOL": torch.bool,
        }
        # add float8 types if available
        if hasattr(torch, "float8_e5m2"):
            dtype_map["F8_E5M2"] = torch.float8_e5m2
        if hasattr(torch, "float8_e4m3fn"):
            dtype_map["F8_E4M3"] = torch.float8_e4m3fn
        return dtype_map.get(dtype_str)

    @staticmethod
    def _convert_float8(byte_tensor, dtype_str, shape):
        if dtype_str == "F8_E5M2" and hasattr(torch, "float8_e5m2"):
            return byte_tensor.view(torch.float8_e5m2).reshape(shape)
        elif dtype_str == "F8_E4M3" and hasattr(torch, "float8_e4m3fn"):
            return byte_tensor.view(torch.float8_e4m3fn).reshape(shape)
        else:
            # # convert to float16 if float8 is not supported
            # print(f"Warning: {dtype_str} is not supported in this PyTorch version. Converting to float16.")
            # return byte_tensor.view(torch.uint8).to(torch.float16).reshape(shape)
            raise ValueError(f"Unsupported float8 type: {dtype_str} (upgrade PyTorch to support float8 types)")


def _default_read_threads() -> int:
    try:
        return max(1, int(os.environ.get("H3_LOAD_THREADS", "8")))
    except ValueError:
        return 8


def _should_drop_page_cache(path: str) -> bool:
    env = os.environ.get("H3_LOAD_DROP_CACHE")
    if env is not None:
        return env.strip().lower() not in ("0", "false", "no", "")
    return os.path.getsize(path) >= (1 << 30)


def safetensors_key_count(path: str) -> int:
    """Number of tensors in a safetensors file (header read only)."""
    with MemoryEfficientSafeOpen(path) as f:
        return len(f.keys())


def safetensors_key_index(paths) -> Dict[str, str]:
    """Map tensor key -> containing shard path, reading only the shard headers.

    Deliberately independent of `*.index.json`: `--dit` may name a bare directory of shards or a
    single merged file, neither of which is guaranteed to ship an index.
    """
    if isinstance(paths, str):
        paths = [paths]
    index = {}
    for path in paths:
        with MemoryEfficientSafeOpen(path) as f:
            for key in f.keys():
                index[key] = path
    return index


def stream_safetensors(
    path: str,
    num_threads: Optional[int] = None,
    read_ahead: Optional[int] = None,
    drop_page_cache: Optional[bool] = None,
    keys: Optional[list] = None,
):
    """Yield (key, tensor) in file order, reading with a thread pool (os.preadv, QD ~num_threads).

    drop_page_cache (default: on for files >= 1 GB, override $H3_LOAD_DROP_CACHE=0/1) evicts
    each tensor's range from the kernel page cache after reading, so a model-sized file does
    not double apparent RAM use during load. num_threads defaults to $H3_LOAD_THREADS or 8.

    `keys` restricts the read to a subset, re-sorted into file order so the reads stay sequential.
    """
    if num_threads is None:
        num_threads = _default_read_threads()
    if drop_page_cache is None:
        drop_page_cache = _should_drop_page_cache(path)
    drop_page_cache = drop_page_cache and hasattr(os, "posix_fadvise")
    with MemoryEfficientSafeOpen(path) as f:
        if keys is None:
            keys = f.keys()
        else:
            missing = [k for k in keys if k not in f.header]
            if missing:
                raise KeyError(f"{path} has no tensor '{missing[0]}' ({len(missing)} keys missing)")
            keys = sorted(keys, key=lambda k: f.header[k]["data_offsets"][0])
        data_start = f.header_size + 8
        if num_threads <= 1 or not hasattr(os, "preadv"):
            seq_fd = f.file.fileno()
            for key in keys:
                yield key, f.get_tensor(key)
                if drop_page_cache:
                    offset_start, offset_end = f.header[key]["data_offsets"]
                    if offset_end > offset_start:
                        os.posix_fadvise(
                            seq_fd, data_start + offset_start, offset_end - offset_start, os.POSIX_FADV_DONTNEED
                        )
            return

        fd = os.open(path, os.O_RDONLY)
        try:

            def read_entry(key):
                offset_start, offset_end = f.header[key]["data_offsets"]
                length = offset_end - offset_start
                if length == 0:
                    return None
                buf = bytearray(length)
                view = memoryview(buf)
                pos = 0
                while pos < length:
                    n = os.preadv(fd, [view[pos:]], data_start + offset_start + pos)
                    if n <= 0:
                        raise IOError(f"short read for tensor '{key}' in {path}")
                    pos += n
                if drop_page_cache:
                    os.posix_fadvise(fd, data_start + offset_start, length, os.POSIX_FADV_DONTNEED)
                return buf

            if read_ahead is None:
                read_ahead = num_threads * 4
            with ThreadPoolExecutor(max_workers=num_threads) as pool:
                key_iter = iter(keys)
                pending = deque(
                    (key, pool.submit(read_entry, key)) for key in itertools.islice(key_iter, read_ahead)
                )
                while pending:
                    key, future = pending.popleft()
                    buf = future.result()
                    next_key = next(key_iter, None)
                    if next_key is not None:
                        pending.append((next_key, pool.submit(read_entry, next_key)))
                    yield key, f._deserialize_tensor(buf, f.header[key])
        finally:
            os.close(fd)


def load_safetensors(
    path: str, device: Union[str, torch.device], disable_mmap: bool = False, dtype: Optional[torch.dtype] = None
) -> dict[str, torch.Tensor]:
    # Handle sharded models stored in a directory
    if os.path.isdir(path):
        import glob
        shard_files = sorted(glob.glob(os.path.join(path, "*.safetensors")))
        if not shard_files:
            raise FileNotFoundError(f"No .safetensors files found in directory: {path}")

        total_state_dict = {}
        for shard_file in shard_files:
            # Recursively call this function to load each shard (which is a single file)
            shard_state_dict = load_safetensors(shard_file, device, disable_mmap, dtype)
            total_state_dict.update(shard_state_dict)
        return total_state_dict

    if disable_mmap:
        state_dict = {}
        for key, value in stream_safetensors(path):
            state_dict[key] = value.to(device, dtype=dtype)
        return state_dict
    else:
        try:
            state_dict = load_file(path, device=device)
        except:
            state_dict = load_file(path)  # prevent device invalid Error
        if dtype is not None:
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].to(dtype=dtype)
        return state_dict


def load_split_weights(
    file_path: str, device: Union[str, torch.device] = "cpu", disable_mmap: bool = False
) -> Dict[str, torch.Tensor]:
    """
    Load split weights from a file. If the file name ends with 00001-of-00004 etc, it will load all files with the same prefix.
    dtype is as is, no conversion is done.
    """
    device = torch.device(device)

    # if the file name ends with 00001-of-00004 etc, we need to load the files with the same prefix
    basename = os.path.basename(file_path)
    match = re.match(r"^(.*?)(\d+)-of-(\d+)\.safetensors$", basename)
    if match:
        prefix = basename[: match.start(2)]
        count = int(match.group(3))
        state_dict = {}
        for i in range(count):
            filename = f"{prefix}{i+1:05d}-of-{count:05d}.safetensors"
            filepath = os.path.join(os.path.dirname(file_path), filename)
            if os.path.exists(filepath):
                state_dict.update(load_safetensors(filepath, device=device, disable_mmap=disable_mmap))
            else:
                raise FileNotFoundError(f"File {filepath} not found")
    else:
        state_dict = load_safetensors(file_path, device=device, disable_mmap=disable_mmap)
    return state_dict
