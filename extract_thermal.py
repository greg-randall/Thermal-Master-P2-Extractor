#!/usr/bin/env python3
"""
Thermal Image Frame Extractor
Extracts embedded frames and metadata from USB_IR_RS300_P2L thermal camera JPEG files.

Usage:
    python extract_thermal.py <image.jpg> [--all]

Options:
    --all    Extract all 52 frames, including empty ones
"""

import argparse
import os
import struct
import sys
from pathlib import Path
from datetime import datetime


def raw_to_kelvin(raw):
    """Convert raw sensor value to Kelvin."""
    return raw / 64


def raw_to_celsius(raw):
    """Convert raw sensor value to Celsius."""
    return raw / 64 - 273.15


def raw_to_fahrenheit(raw):
    """Convert raw sensor value to Fahrenheit."""
    return raw * 0.028125 - 459.67


def format_temp(raw, use_celsius=False):
    """Format temperature with unit."""
    if use_celsius:
        return f"{raw_to_celsius(raw):.1f}°C"
    else:
        return f"{raw_to_fahrenheit(raw):.1f}°F"


def parse_jpeg_markers(data: bytes) -> dict:
    """Parse JPEG markers and return structured data."""
    markers = {
        'app1_exif': None,
        'app2_ijpeg': None,
        'app3_frames': [],
        'app4_data': None,
        'app5_calibration': None,
        'app6_data': None,
        'app7_data': None,
        'app8_data': None,
        'app9_info': None,
        'jpeg_start': None,
        'jpeg_end': None,
    }

    i = 0
    while i < len(data) - 3:
        if data[i] != 0xff:
            i += 1
            continue

        marker = data[i + 1]

        # Start of Image
        if marker == 0xd8:
            i += 2
            continue

        # End of Image
        if marker == 0xd9:
            markers['jpeg_end'] = i
            break

        # Skip stuffed bytes
        if marker == 0x00 or marker == 0xff:
            i += 1
            continue

        # Markers with length field
        if i + 4 > len(data):
            break

        length = (data[i + 2] << 8) | data[i + 3]
        content = data[i + 4:i + 4 + length - 2]

        if marker == 0xe1:  # APP1 - EXIF
            markers['app1_exif'] = content
        elif marker == 0xe2:  # APP2 - IJPEG or ICC
            if content.startswith(b'ICC_PROFILE'):
                # This is the ICC profile before the main JPEG
                pass
            else:
                markers['app2_ijpeg'] = content
        elif marker == 0xe3:  # APP3 - Frame data
            markers['app3_frames'].append(content)
        elif marker == 0xe4:  # APP4
            markers['app4_data'] = content
        elif marker == 0xe5:  # APP5 - Calibration
            markers['app5_calibration'] = content
        elif marker == 0xe6:  # APP6
            markers['app6_data'] = content
        elif marker == 0xe7:  # APP7
            markers['app7_data'] = content
        elif marker == 0xe8:  # APP8
            markers['app8_data'] = content
        elif marker == 0xe9:  # APP9 - Info string
            markers['app9_info'] = content
        elif marker == 0xe0:  # APP0 - JFIF (start of main JPEG)
            markers['jpeg_start'] = i
            # Don't advance past this - we want to capture the whole JPEG
            break

        i += 2 + length

    return markers


def parse_exif(exif_data: bytes) -> dict:
    """Parse EXIF data and extract useful fields."""
    info = {}

    if not exif_data or not exif_data.startswith(b'Exif\x00\x00'):
        return info

    tiff_data = exif_data[6:]

    # Determine byte order
    if tiff_data[:2] == b'MM':
        byte_order = '>'  # Big-endian
    elif tiff_data[:2] == b'II':
        byte_order = '<'  # Little-endian
    else:
        return info

    info['byte_order'] = 'Big-endian (Motorola)' if byte_order == '>' else 'Little-endian (Intel)'

    # Parse IFD entries
    try:
        ifd_offset = struct.unpack(byte_order + 'I', tiff_data[4:8])[0]
        num_entries = struct.unpack(byte_order + 'H', tiff_data[ifd_offset:ifd_offset + 2])[0]

        tag_names = {
            0x010f: 'Make',
            0x0110: 'Model',
            0x010e: 'ImageDescription',
            0x0132: 'DateTime',
            0x0100: 'ImageWidth',
            0x0101: 'ImageHeight',
            0x0112: 'Orientation',
        }

        for entry_idx in range(num_entries):
            entry_offset = ifd_offset + 2 + entry_idx * 12
            tag = struct.unpack(byte_order + 'H', tiff_data[entry_offset:entry_offset + 2])[0]
            dtype = struct.unpack(byte_order + 'H', tiff_data[entry_offset + 2:entry_offset + 4])[0]
            count = struct.unpack(byte_order + 'I', tiff_data[entry_offset + 4:entry_offset + 8])[0]
            value_offset = entry_offset + 8

            if tag in tag_names:
                tag_name = tag_names[tag]

                if dtype == 2:  # ASCII string
                    if count > 4:
                        str_offset = struct.unpack(byte_order + 'I', tiff_data[value_offset:value_offset + 4])[0]
                        value = tiff_data[str_offset:str_offset + count - 1].decode('ascii', errors='ignore')
                    else:
                        value = tiff_data[value_offset:value_offset + count - 1].decode('ascii', errors='ignore')
                    info[tag_name] = value
                elif dtype == 3:  # SHORT
                    value = struct.unpack(byte_order + 'H', tiff_data[value_offset:value_offset + 2])[0]
                    info[tag_name] = value
                elif dtype == 4:  # LONG
                    value = struct.unpack(byte_order + 'I', tiff_data[value_offset:value_offset + 4])[0]
                    info[tag_name] = value
    except Exception as e:
        info['parse_error'] = str(e)

    return info


def parse_calibration(cal_data: bytes) -> dict:
    """Parse calibration data from APP5."""
    cal = {}

    if not cal_data:
        return cal

    try:
        # Parse as little-endian floats
        num_floats = len(cal_data) // 4
        floats = struct.unpack('<' + 'f' * num_floats, cal_data[:num_floats * 4])

        cal['raw_floats'] = list(floats)

        # Known calibration fields (based on typical thermal camera formats)
        if len(floats) >= 6:
            cal['ambient_temperature'] = floats[0]
            cal['emissivity_factor'] = floats[1]
            cal['distance_factor'] = floats[2]
            cal['humidity_factor'] = floats[3]
            cal['reflected_temperature'] = floats[4]
    except Exception as e:
        cal['parse_error'] = str(e)

    return cal


def parse_app4_data(app4_data: bytes) -> dict:
    """Parse APP4 data which may contain additional sensor info."""
    info = {}

    if not app4_data:
        return info

    try:
        info['size'] = len(app4_data)
        # First few bytes often contain resolution/format info
        if len(app4_data) >= 8:
            info['header_bytes'] = ' '.join(f'{b:02x}' for b in app4_data[:16])
    except Exception as e:
        info['parse_error'] = str(e)

    return info


def parse_app9_info(app9_data: bytes) -> dict:
    """Parse APP9 data containing device/software info."""
    info = {}

    if not app9_data:
        return info

    try:
        # Usually contains null-terminated strings
        text = app9_data.split(b'\x00')[0].decode('ascii', errors='ignore')
        info['software'] = text
        info['raw_size'] = len(app9_data)
    except Exception as e:
        info['parse_error'] = str(e)

    return info


def extract_frames(app3_segments: list, width: int = 256, height: int = 192) -> list:
    """Extract frame data from APP3 segments."""
    # Combine all APP3 data
    combined = bytearray()
    for segment in app3_segments:
        combined.extend(segment)

    pixels = width * height
    frame_size = pixels * 2  # 16-bit per pixel
    num_frames = len(combined) // frame_size

    frames = []
    for i in range(min(num_frames, 52)):  # Max 52 frames
        offset = i * frame_size
        frame_data = bytes(combined[offset:offset + frame_size])

        # Check if frame is empty (all zeros)
        is_empty = all(b == 0 for b in frame_data)

        frames.append({
            'index': i,
            'data': frame_data,
            'is_empty': is_empty,
            'size': len(frame_data),
        })

    return frames


def get_image_extension(fmt: str) -> str:
    """Get the proper file extension for an image format."""
    extensions = {
        'png': '.png',
        'jpg': '.jpg',
        'jpeg': '.jpg',
        'tif': '.tif',
        'tiff': '.tif',
        'bmp': '.bmp',
        'webp': '.webp',
    }
    return extensions.get(fmt.lower(), '.png')


def save_preview_frame(frame_data: bytes, output_path: Path, width: int = 256, height: int = 192, fmt: str = 'png'):
    """Save frame 00 as grayscale preview image."""
    try:
        import numpy as np
        from PIL import Image

        # Frame 00 is encoded as pairs: grayscale_byte, 0x80
        low_bytes = np.array([frame_data[i] for i in range(0, len(frame_data), 2)], dtype=np.uint8)
        preview = low_bytes.reshape((height, width))

        img = Image.fromarray(preview, mode='L')

        # Update extension based on format
        output_path = output_path.with_suffix(get_image_extension(fmt))

        # Handle format-specific options
        if fmt.lower() in ['jpg', 'jpeg']:
            img.save(output_path, quality=95)
        else:
            img.save(output_path)

        return True, int(preview.min()), int(preview.max()), output_path
    except ImportError:
        # Save raw if PIL not available
        raw_path = output_path.with_suffix('.bin')
        with open(raw_path, 'wb') as f:
            f.write(frame_data)
        return False, None, None, raw_path


def save_thermal_frame(frame_data: bytes, output_path: Path, width: int = 256, height: int = 192,
                       fmt: str = 'png', save_raw: bool = False,
                       use_clahe: bool = False, clahe_clip: float = 2.0, clahe_grid: int = 8):
    """Save raw thermal frame. TIFF preserves 16-bit, others get normalized to 8-bit."""
    try:
        import numpy as np
        from PIL import Image

        # 16-bit little-endian
        raw_array = np.frombuffer(frame_data, dtype='<u2').reshape((height, width))

        raw_min, raw_max = int(raw_array.min()), int(raw_array.max())
        raw_mean = float(raw_array.mean())
        raw_std = float(raw_array.std())

        # Calculate temperatures
        temp_min_c = raw_to_celsius(raw_min)
        temp_max_c = raw_to_celsius(raw_max)
        temp_mean_c = raw_to_celsius(raw_mean)
        temp_min_f = raw_to_fahrenheit(raw_min)
        temp_max_f = raw_to_fahrenheit(raw_max)
        temp_mean_f = raw_to_fahrenheit(raw_mean)

        # Update extension based on format
        output_path = output_path.with_suffix(get_image_extension(fmt))

        # TIFF can preserve 16-bit data
        if fmt.lower() in ['tif', 'tiff']:
            img = Image.fromarray(raw_array, mode='I;16')
            img.save(output_path)
        else:
            # Normalize to 8-bit for other formats
            if raw_max > raw_min:
                normalized = ((raw_array - raw_min) / (raw_max - raw_min) * 255).astype(np.uint8)
            else:
                normalized = np.zeros((height, width), dtype=np.uint8)

            # Apply CLAHE if requested
            if use_clahe:
                try:
                    import cv2
                    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_grid, clahe_grid))
                    normalized = clahe.apply(normalized)
                except ImportError:
                    print("    Warning: OpenCV not available, skipping CLAHE")

            img = Image.fromarray(normalized, mode='L')
            if fmt.lower() in ['jpg', 'jpeg']:
                img.save(output_path, quality=95)
            else:
                img.save(output_path)

        # Optionally save raw 16-bit as numpy file
        if save_raw:
            npy_path = output_path.with_name(output_path.stem + '_raw.npy')
            np.save(npy_path, raw_array)

        return True, {
            'min': raw_min, 'max': raw_max, 'mean': raw_mean, 'std': raw_std,
            'temp_min_c': temp_min_c, 'temp_max_c': temp_max_c, 'temp_mean_c': temp_mean_c,
            'temp_min_f': temp_min_f, 'temp_max_f': temp_max_f, 'temp_mean_f': temp_mean_f,
            'clahe': use_clahe
        }, output_path
    except ImportError:
        # Save raw binary if numpy not available
        raw_path = output_path.with_suffix('.bin')
        with open(raw_path, 'wb') as f:
            f.write(frame_data)
        return False, None, raw_path


def save_empty_frame(frame_data: bytes, output_path: Path, width: int = 256, height: int = 192, fmt: str = 'png'):
    """Save an empty frame as a black image."""
    try:
        import numpy as np
        from PIL import Image

        # Update extension based on format
        output_path = output_path.with_suffix(get_image_extension(fmt))

        black = np.zeros((height, width), dtype=np.uint8)
        img = Image.fromarray(black, mode='L')

        if fmt.lower() in ['jpg', 'jpeg']:
            img.save(output_path, quality=95)
        else:
            img.save(output_path)

        return True, output_path
    except ImportError:
        return False, None


def extract_main_jpeg(data: bytes, jpeg_start: int) -> bytes:
    """Extract the main JPEG image from the file."""
    # Find the end of the JPEG (FFD9)
    for i in range(len(data) - 1, jpeg_start, -1):
        if data[i - 1] == 0xff and data[i] == 0xd9:
            jpeg_data = data[jpeg_start:i + 1]
            # Prepend SOI marker if missing (jpeg_start points to APP0, not SOI)
            if not jpeg_data.startswith(b'\xff\xd8'):
                jpeg_data = b'\xff\xd8' + jpeg_data
            return jpeg_data
    # Fallback: return from jpeg_start to end with SOI
    jpeg_data = data[jpeg_start:]
    if not jpeg_data.startswith(b'\xff\xd8'):
        jpeg_data = b'\xff\xd8' + jpeg_data
    return jpeg_data


def write_metadata_report(output_path: Path, metadata: dict):
    """Write metadata report to text file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("THERMAL IMAGE EXTRACTION REPORT\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Source File: {metadata.get('source_file', 'Unknown')}\n")
        f.write(f"File Size: {metadata.get('file_size', 0):,} bytes\n")
        f.write(f"Extraction Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # EXIF Information
        f.write("-" * 70 + "\n")
        f.write("EXIF INFORMATION\n")
        f.write("-" * 70 + "\n")
        exif = metadata.get('exif', {})
        if exif:
            for key, value in exif.items():
                f.write(f"  {key}: {value}\n")
        else:
            f.write("  No EXIF data found\n")
        f.write("\n")

        # Calibration Data
        f.write("-" * 70 + "\n")
        f.write("CALIBRATION DATA\n")
        f.write("-" * 70 + "\n")
        cal = metadata.get('calibration', {})
        if cal:
            if 'ambient_temperature' in cal:
                f.write(f"  Ambient Temperature: {cal['ambient_temperature']}\n")
            if 'emissivity_factor' in cal:
                f.write(f"  Emissivity Factor: {cal['emissivity_factor']}\n")
            if 'distance_factor' in cal:
                f.write(f"  Distance Factor: {cal['distance_factor']}\n")
            if 'humidity_factor' in cal:
                f.write(f"  Humidity Factor: {cal['humidity_factor']}\n")
            if 'reflected_temperature' in cal:
                f.write(f"  Reflected Temperature: {cal['reflected_temperature']}\n")
            if 'raw_floats' in cal:
                f.write(f"  Raw Float Values: {cal['raw_floats']}\n")
        else:
            f.write("  No calibration data found\n")
        f.write("\n")

        # Device/Software Info
        f.write("-" * 70 + "\n")
        f.write("DEVICE/SOFTWARE INFORMATION\n")
        f.write("-" * 70 + "\n")
        app9 = metadata.get('app9_info', {})
        if app9:
            for key, value in app9.items():
                f.write(f"  {key}: {value}\n")
        else:
            f.write("  No device info found\n")
        f.write("\n")

        # Frame Information
        f.write("-" * 70 + "\n")
        f.write("FRAME INFORMATION\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Frame Dimensions: {metadata.get('frame_width', 256)} x {metadata.get('frame_height', 192)}\n")
        f.write(f"  Total Frame Slots: {metadata.get('total_frames', 0)}\n")
        f.write(f"  Non-Empty Frames: {metadata.get('non_empty_frames', 0)}\n")
        f.write(f"  Frames Extracted: {metadata.get('frames_extracted', 0)}\n\n")

        # Frame Details
        frame_details = metadata.get('frame_details', [])
        if frame_details:
            f.write("  Frame Details:\n")
            for fd in frame_details:
                status = "EMPTY" if fd.get('is_empty') else "DATA"
                f.write(f"    Frame {fd['index']:02d}: [{status}]")
                if not fd.get('is_empty'):
                    if 'stats' in fd:
                        stats = fd['stats']
                        f.write(f" min={stats['min']}, max={stats['max']}, mean={stats['mean']:.1f}, std={stats['std']:.1f}")
                    elif 'preview_range' in fd:
                        f.write(f" grayscale range={fd['preview_range'][0]}-{fd['preview_range'][1]}")
                f.write("\n")
        f.write("\n")

        # Temperature Readings
        use_celsius = metadata.get('use_celsius', False)
        temp_unit = "°C" if use_celsius else "°F"
        f.write("-" * 70 + "\n")
        f.write("TEMPERATURE READINGS\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Unit: {'Celsius' if use_celsius else 'Fahrenheit'}\n")
        f.write(f"  Conversion: T(K) = raw / 64\n\n")

        for fd in frame_details:
            if 'stats' in fd:
                stats = fd['stats']
                if use_celsius:
                    t_min = stats.get('temp_min_c', 0)
                    t_max = stats.get('temp_max_c', 0)
                    t_mean = stats.get('temp_mean_c', 0)
                else:
                    t_min = stats.get('temp_min_f', 0)
                    t_max = stats.get('temp_max_f', 0)
                    t_mean = stats.get('temp_mean_f', 0)

                f.write(f"  Frame {fd['index']:02d} (Thermal):\n")
                f.write(f"    Minimum:  {t_min:.1f}{temp_unit}\n")
                f.write(f"    Maximum:  {t_max:.1f}{temp_unit}\n")
                f.write(f"    Average:  {t_mean:.1f}{temp_unit}\n")
                f.write(f"    Span:     {t_max - t_min:.1f}{temp_unit}\n\n")
        f.write("")

        # Image Processing Notes
        f.write("-" * 70 + "\n")
        f.write("IMAGE PROCESSING NOTES\n")
        f.write("-" * 70 + "\n")
        out_fmt = metadata.get('output_format', 'png').lower()
        use_clahe = metadata.get('use_clahe', False)
        f.write(f"  Output Format: {out_fmt.upper()}\n")
        f.write(f"  Raw Files Saved: {'Yes' if metadata.get('save_raw') else 'No'}\n")
        f.write(f"  CLAHE Applied: {'Yes' if use_clahe else 'No'}\n\n")

        if out_fmt in ['tif', 'tiff']:
            f.write("  Thermal Frame Processing:\n")
            f.write("    - Full 16-bit radiometric data preserved (no manipulation)\n")
            f.write("    - Pixel values are raw sensor ADC counts\n")
        else:
            f.write("  Thermal Frame Processing:\n")
            f.write("    Step 1 - Normalization:\n")
            f.write("      - Normalized from 16-bit to 8-bit using min-max linear stretch\n")
            f.write("      - Formula: output = (pixel - min) / (max - min) * 255\n")
            f.write("      - Min/max values auto-detected from each frame's pixel data\n")
            f.write("      - Coldest pixel maps to 0 (black), hottest to 255 (white)\n")
            # Add actual values used if available
            for fd in frame_details:
                if 'stats' in fd:
                    stats = fd['stats']
                    f.write(f"      - Frame {fd['index']:02d} stretched using min={stats['min']}, max={stats['max']}\n")

            if use_clahe:
                f.write("\n    Step 2 - CLAHE (Contrast Limited Adaptive Histogram Equalization):\n")
                f.write(f"      - Clip Limit: {metadata.get('clahe_clip', 2.0)}\n")
                f.write(f"      - Grid Size: {metadata.get('clahe_grid', 8)}x{metadata.get('clahe_grid', 8)}\n")
                f.write("      - Enhances local contrast while limiting noise amplification\n")
                f.write("      - Applied after normalization to 8-bit\n")

            f.write("\n    WARNING: This processing loses radiometric precision\n")

        f.write("\n  Preview Frame Processing:\n")
        f.write("    - Extracted 8-bit grayscale from packed format (low byte of each 16-bit word)\n")
        f.write("    - No histogram manipulation applied\n")
        f.write("\n")

        # Main JPEG Info
        f.write("-" * 70 + "\n")
        f.write("MAIN JPEG IMAGE\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Size: {metadata.get('jpeg_size', 0):,} bytes\n")
        f.write(f"  Extracted: {metadata.get('jpeg_extracted', False)}\n\n")

        # APP Segment Summary
        f.write("-" * 70 + "\n")
        f.write("APP SEGMENT SUMMARY\n")
        f.write("-" * 70 + "\n")
        segments = metadata.get('app_segments', {})
        for seg_name, seg_size in segments.items():
            f.write(f"  {seg_name}: {seg_size:,} bytes\n")
        f.write("\n")

        f.write("=" * 70 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Extract frames and metadata from USB_IR_RS300_P2L thermal camera JPEG files.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python extract_thermal.py image.jpg                    Extract non-empty frames as PNG
  python extract_thermal.py image.jpg --all              Extract all 52 frames
  python extract_thermal.py image.jpg --format tif       Extract as 16-bit TIFF (preserves raw data)
  python extract_thermal.py image.jpg --format jpg       Extract as JPEG

Supported formats:
  png   - PNG (default, lossless 8-bit)
  tif   - TIFF (16-bit, preserves full radiometric data for thermal frames)
  jpg   - JPEG (lossy 8-bit)
  bmp   - BMP (lossless 8-bit)
  webp  - WebP (lossless 8-bit)
        """
    )
    parser.add_argument('image', help='Input thermal JPEG image file')
    parser.add_argument('--all', action='store_true', help='Extract all frames including empty ones')
    parser.add_argument('--format', '-f', type=str, default='png',
                        choices=['png', 'tif', 'tiff', 'jpg', 'jpeg', 'bmp', 'webp'],
                        help='Output image format (default: png). Use tif for 16-bit thermal data.')
    parser.add_argument('--clahe', action='store_true',
                        help='Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to thermal frames')
    parser.add_argument('--clahe-clip', type=float, default=2.0,
                        help='CLAHE clip limit (default: 2.0). Higher = more contrast.')
    parser.add_argument('--clahe-grid', type=int, default=8,
                        help='CLAHE grid size (default: 8). Smaller = more local contrast.')
    parser.add_argument('--save-raw', action='store_true', help='Also save raw .bin and .npy files')
    parser.add_argument('--celsius', action='store_true', help='Display temperatures in Celsius (default: Fahrenheit)')
    parser.add_argument('--width', type=int, default=256, help='Frame width (default: 256)')
    parser.add_argument('--height', type=int, default=192, help='Frame height (default: 192)')

    args = parser.parse_args()

    # Validate input file
    input_path = Path(args.image)
    if not input_path.exists():
        print(f"Error: File not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    if not input_path.suffix.lower() in ['.jpg', '.jpeg']:
        print(f"Warning: File does not have .jpg extension: {input_path}", file=sys.stderr)

    # Create output directory
    output_dir = input_path.parent / input_path.stem
    output_dir.mkdir(exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Read input file
    print(f"Reading: {input_path}")
    with open(input_path, 'rb') as f:
        data = f.read()

    print(f"File size: {len(data):,} bytes")

    # Parse JPEG markers
    print("Parsing JPEG structure...")
    markers = parse_jpeg_markers(data)

    # Collect metadata
    metadata = {
        'source_file': str(input_path.absolute()),
        'file_size': len(data),
        'frame_width': args.width,
        'frame_height': args.height,
        'output_format': args.format,
        'save_raw': args.save_raw,
        'use_clahe': args.clahe,
        'clahe_clip': args.clahe_clip,
        'clahe_grid': args.clahe_grid,
        'use_celsius': args.celsius,
        'app_segments': {},
    }

    # Parse EXIF
    if markers['app1_exif']:
        metadata['exif'] = parse_exif(markers['app1_exif'])
        metadata['app_segments']['APP1 (EXIF)'] = len(markers['app1_exif'])
        print(f"  Found EXIF data: {len(markers['app1_exif'])} bytes")

    # Parse calibration
    if markers['app5_calibration']:
        metadata['calibration'] = parse_calibration(markers['app5_calibration'])
        metadata['app_segments']['APP5 (Calibration)'] = len(markers['app5_calibration'])
        print(f"  Found calibration data: {len(markers['app5_calibration'])} bytes")

    # Parse APP9 info
    if markers['app9_info']:
        metadata['app9_info'] = parse_app9_info(markers['app9_info'])
        metadata['app_segments']['APP9 (Info)'] = len(markers['app9_info'])
        print(f"  Found device info: {len(markers['app9_info'])} bytes")

    # Parse other APP segments
    if markers['app2_ijpeg']:
        metadata['app_segments']['APP2 (IJPEG)'] = len(markers['app2_ijpeg'])
    if markers['app4_data']:
        metadata['app_segments']['APP4'] = len(markers['app4_data'])
    if markers['app6_data']:
        metadata['app_segments']['APP6'] = len(markers['app6_data'])
    if markers['app7_data']:
        metadata['app_segments']['APP7'] = len(markers['app7_data'])
    if markers['app8_data']:
        metadata['app_segments']['APP8'] = len(markers['app8_data'])

    # Extract frames
    if not markers['app3_frames']:
        print("  WARNING: No APP3 segments found - this file has no embedded thermal data")
        print("           Only the visible JPEG image will be extracted")
        metadata['has_thermal_data'] = False
        frames = []
        non_empty = []
    else:
        print(f"  Found {len(markers['app3_frames'])} APP3 segments")
        total_app3_size = sum(len(s) for s in markers['app3_frames'])
        metadata['app_segments']['APP3 (Frames)'] = total_app3_size
        metadata['has_thermal_data'] = True

        frames = extract_frames(markers['app3_frames'], args.width, args.height)
        non_empty = [f for f in frames if not f['is_empty']]
        print(f"  Total frames: {len(frames)}, Non-empty: {len(non_empty)}")

    metadata['total_frames'] = len(frames)
    metadata['non_empty_frames'] = len(non_empty)

    # Determine which frames to extract
    frames_to_extract = frames if args.all else non_empty
    metadata['frames_extracted'] = len(frames_to_extract)

    # Extract frames
    print(f"\nExtracting {'all' if args.all else 'non-empty'} frames...")
    metadata['frame_details'] = []

    img_ext = get_image_extension(args.format)
    for frame in frames_to_extract:
        frame_info = {'index': frame['index'], 'is_empty': frame['is_empty']}

        if frame['is_empty']:
            output_path = output_dir / f"frame_{frame['index']:02d}_empty{img_ext}"
            success, actual_path = save_empty_frame(frame['data'], output_path, args.width, args.height, args.format)
            print(f"  Frame {frame['index']:02d}: empty" + (" (saved)" if success else " (save failed)"))
        elif frame['index'] == 0:
            # Preview frame
            output_path = output_dir / f"frame_{frame['index']:02d}_preview{img_ext}"
            success, min_val, max_val, actual_path = save_preview_frame(frame['data'], output_path, args.width, args.height, args.format)
            if success:
                frame_info['preview_range'] = (min_val, max_val)
                print(f"  Frame {frame['index']:02d}: preview (grayscale {min_val}-{max_val})")
            else:
                print(f"  Frame {frame['index']:02d}: preview (saved as raw binary)")
        else:
            # Thermal data frame
            output_path = output_dir / f"frame_{frame['index']:02d}_thermal{img_ext}"
            success, stats, actual_path = save_thermal_frame(
                frame['data'], output_path, args.width, args.height, args.format, args.save_raw,
                args.clahe, args.clahe_clip, args.clahe_grid
            )
            if success and stats:
                frame_info['stats'] = stats
                if args.format.lower() in ['tif', 'tiff']:
                    bit_info = "16-bit"
                elif args.clahe:
                    bit_info = "8-bit CLAHE"
                else:
                    bit_info = "8-bit normalized"
                # Show temperature range
                if args.celsius:
                    temp_range = f"{stats['temp_min_c']:.1f}°C to {stats['temp_max_c']:.1f}°C"
                else:
                    temp_range = f"{stats['temp_min_f']:.1f}°F to {stats['temp_max_f']:.1f}°F"
                print(f"  Frame {frame['index']:02d}: thermal ({bit_info}, {temp_range})")
            else:
                print(f"  Frame {frame['index']:02d}: thermal (saved as raw binary)")

        # Optionally save raw binary for all non-empty frames
        if args.save_raw and not frame['is_empty']:
            raw_path = output_dir / f"frame_{frame['index']:02d}_raw.bin"
            with open(raw_path, 'wb') as f:
                f.write(frame['data'])

        metadata['frame_details'].append(frame_info)

    # Extract main JPEG
    print("\nExtracting main JPEG image...")
    if markers['jpeg_start']:
        jpeg_data = extract_main_jpeg(data, markers['jpeg_start'])
        jpeg_path = output_dir / f"{input_path.stem}_visible.jpg"
        with open(jpeg_path, 'wb') as f:
            f.write(jpeg_data)
        metadata['jpeg_size'] = len(jpeg_data)
        metadata['jpeg_extracted'] = True
        print(f"  Saved: {jpeg_path.name} ({len(jpeg_data):,} bytes)")
    else:
        metadata['jpeg_size'] = 0
        metadata['jpeg_extracted'] = False
        print("  Warning: Could not locate main JPEG image")

    # Write metadata report
    print("\nWriting metadata report...")
    report_path = output_dir / f"{input_path.stem}_metadata.txt"
    write_metadata_report(report_path, metadata)
    print(f"  Saved: {report_path.name}")

    # Summary
    print("\n" + "=" * 50)
    print("EXTRACTION COMPLETE")
    print("=" * 50)
    print(f"Output directory: {output_dir}")
    print(f"Files created:")
    for f in sorted(output_dir.iterdir()):
        print(f"  {f.name} ({f.stat().st_size:,} bytes)")


if __name__ == '__main__':
    main()
