/// FicTrac http://rjdmoore.net/fictrac/
/// \file       SharedMemoryPublisher.cpp
/// \brief      Publishes remapped frames and tracking data to shared memory.

#include "SharedMemoryPublisher.h"
#include "Logger.h"

#include <windows.h>
#include <cstring>

///
/// Create named shared memory regions for frame and tracking data.
///
SharedMemoryPublisher::SharedMemoryPublisher(
    const std::string& frame_shm_name,
    int frame_w, int frame_h, int frame_slots,
    const std::string& tracking_shm_name, int tracking_slots)
    : _frame_w(frame_w), _frame_h(frame_h), _frame_slots(frame_slots),
      _tracking_slots(tracking_slots), _active(false),
      _frame_shm_handle(nullptr), _frame_shm_ptr(nullptr),
      _tracking_shm_handle(nullptr), _tracking_shm_ptr(nullptr)
{
    _frame_bytes = static_cast<size_t>(frame_w) * frame_h;
    _frame_shm_size = HEADER_SIZE + _frame_slots * _frame_bytes;
    _tracking_shm_size = HEADER_SIZE + _tracking_slots * sizeof(TrackingSlot);

    // Create frame shared memory
    _frame_shm_handle = CreateFileMappingA(
        INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE,
        0, static_cast<DWORD>(_frame_shm_size),
        frame_shm_name.c_str());
    if (_frame_shm_handle) {
        _frame_shm_ptr = static_cast<uint8_t*>(
            MapViewOfFile(_frame_shm_handle, FILE_MAP_ALL_ACCESS,
                          0, 0, _frame_shm_size));
        if (_frame_shm_ptr) {
            memset(_frame_shm_ptr, 0, _frame_shm_size);
        }
    }
    if (!_frame_shm_ptr) {
        LOG_ERR("Failed to create frame shared memory '%s' (%lu bytes)",
                frame_shm_name.c_str(), (unsigned long)_frame_shm_size);
        return;
    }

    // Create tracking shared memory
    _tracking_shm_handle = CreateFileMappingA(
        INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE,
        0, static_cast<DWORD>(_tracking_shm_size),
        tracking_shm_name.c_str());
    if (_tracking_shm_handle) {
        _tracking_shm_ptr = static_cast<uint8_t*>(
            MapViewOfFile(_tracking_shm_handle, FILE_MAP_ALL_ACCESS,
                          0, 0, _tracking_shm_size));
        if (_tracking_shm_ptr) {
            memset(_tracking_shm_ptr, 0, _tracking_shm_size);
        }
    }
    if (!_tracking_shm_ptr) {
        LOG_ERR("Failed to create tracking shared memory '%s' (%lu bytes)",
                tracking_shm_name.c_str(), (unsigned long)_tracking_shm_size);
        return;
    }

    _active = true;
    LOG("Shared memory output active:");
    LOG("  frames  = '%s' (%dx%d, %d slots, %lu bytes)",
        frame_shm_name.c_str(), frame_w, frame_h, frame_slots,
        (unsigned long)_frame_shm_size);
    LOG("  tracking = '%s' (%d slots, %lu bytes)",
        tracking_shm_name.c_str(), tracking_slots,
        (unsigned long)_tracking_shm_size);
}

///
///
///
SharedMemoryPublisher::~SharedMemoryPublisher()
{
    if (_frame_shm_ptr) {
        UnmapViewOfFile(_frame_shm_ptr);
    }
    if (_frame_shm_handle) {
        CloseHandle(_frame_shm_handle);
    }
    if (_tracking_shm_ptr) {
        UnmapViewOfFile(_tracking_shm_ptr);
    }
    if (_tracking_shm_handle) {
        CloseHandle(_tracking_shm_handle);
    }
}

///
/// Write a grayscale frame to the next slot in the frame ring buffer.
///
void SharedMemoryPublisher::writeFrame(const cv::Mat& remap_grey)
{
    if (!_active || remap_grey.empty()) { return; }

    volatile long long* wr_idx = reinterpret_cast<volatile long long*>(_frame_shm_ptr);
    uint64_t idx = static_cast<uint64_t>(*wr_idx);
    size_t slot = idx % _frame_slots;
    uint8_t* dst = _frame_shm_ptr + HEADER_SIZE + slot * _frame_bytes;

    // Copy frame row-by-row in case cv::Mat is not contiguous
    if (remap_grey.isContinuous()) {
        memcpy(dst, remap_grey.data, _frame_bytes);
    } else {
        for (int r = 0; r < _frame_h; r++) {
            memcpy(dst + r * _frame_w, remap_grey.ptr(r), _frame_w);
        }
    }

    // Atomic increment (release semantics)
    _InterlockedIncrement64(wr_idx);
}

///
/// Write tracking data to the next slot in the tracking ring buffer.
///
void SharedMemoryPublisher::writeTrackingData(const TrackingSlot& slot_data)
{
    if (!_active) { return; }

    volatile long long* wr_idx = reinterpret_cast<volatile long long*>(_tracking_shm_ptr);
    uint64_t idx = static_cast<uint64_t>(*wr_idx);
    size_t slot = idx % _tracking_slots;
    TrackingSlot* dst = reinterpret_cast<TrackingSlot*>(
        _tracking_shm_ptr + HEADER_SIZE + slot * sizeof(TrackingSlot));

    memcpy(dst, &slot_data, sizeof(TrackingSlot));

    _InterlockedIncrement64(wr_idx);
}
