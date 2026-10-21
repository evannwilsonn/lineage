import CJPEG
import Foundation

/// An 8-bit RGB image, rows top to bottom, pixels interleaved.
public struct RGBImage {
    public var width: Int
    public var height: Int
    public var pixels: [UInt8]
    public init(width: Int, height: Int, pixels: [UInt8]) {
        self.width = width; self.height = height; self.pixels = pixels
    }
}

public enum JPEGError: Error { case unreadable(String) }

/// Decodes a JPEG with libjpeg(-turbo) through its C API.
public func decodeJPEG(_ data: Data) throws -> RGBImage {
    guard data.count > 4, data[data.startIndex] == 0xFF, data[data.startIndex + 1] == 0xD8 else {
        throw JPEGError.unreadable("not a JPEG")
    }
    let cinfo = UnsafeMutablePointer<jpeg_decompress_struct>.allocate(capacity: 1)
    let jerr = UnsafeMutablePointer<jpeg_error_mgr>.allocate(capacity: 1)
    defer { cinfo.deallocate(); jerr.deallocate() }
    cinfo.initialize(to: jpeg_decompress_struct())
    jerr.initialize(to: jpeg_error_mgr())
    cinfo.pointee.err = jpeg_std_error(jerr)
    jpeg_CreateDecompress(cinfo, JPEG_LIB_VERSION, MemoryLayout<jpeg_decompress_struct>.size)
    defer { jpeg_destroy_decompress(cinfo) }
    return data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) -> RGBImage in
        jpeg_mem_src(cinfo, raw.bindMemory(to: UInt8.self).baseAddress, UInt(raw.count))
        jpeg_read_header(cinfo, 1)
        cinfo.pointee.out_color_space = JCS_RGB
        jpeg_start_decompress(cinfo)
        let w = Int(cinfo.pointee.output_width), h = Int(cinfo.pointee.output_height)
        var pixels = [UInt8](repeating: 0, count: w * h * 3)
        pixels.withUnsafeMutableBufferPointer { buf in
            var row: UnsafeMutablePointer<UInt8>? = nil
            while cinfo.pointee.output_scanline < cinfo.pointee.output_height {
                row = buf.baseAddress! + Int(cinfo.pointee.output_scanline) * w * 3
                _ = withUnsafeMutablePointer(to: &row) { jpeg_read_scanlines(cinfo, $0, 1) }
            }
        }
        jpeg_finish_decompress(cinfo)
        return RGBImage(width: w, height: h, pixels: pixels)
    }
}
