/// FicTrac http://rjdmoore.net/fictrac/
/// \file       SharedMemoryPublisher.h
/// \brief      Publishes remapped frames and tracking data to shared memory.

#pragma once

#include <opencv2/core.hpp>

#include <string>
#include <cstdint>

// Forward-declare Trackball::DATA to avoid circular include
class Trackball;

#pragma pack(push, 1)
struct TrackingSlot {
    uint32_t cnt;           //   4 bytes - frame count
    uint32_t seq;           //   4 bytes - sequence number
    double   dr_cam[3];     //  24 bytes - delta rotation (camera frame)
    double   dr_lab[3];     //  24 bytes - delta rotation (lab frame)
    double   r_cam[3];      //  24 bytes - absolute rotation (camera frame)
    double   r_lab[3];      //  24 bytes - absolute rotation (lab frame)
    double   posx;          //   8 bytes - integrated x position
    double   posy;          //   8 bytes - integrated y position
    double   heading;       //   8 bytes - integrated heading
    double   step_dir;      //   8 bytes - direction (radians)
    double   step_mag;      //   8 bytes - speed (radians/frame)
    double   intx;          //   8 bytes - integrated x movement
    double   inty;          //   8 bytes - integrated y movement
    double   ts;            //   8 bytes - timestamp (ms since epoch)
    double   ms;            //   8 bytes - ms since midnight
    double   err;           //   8 bytes - matching error
    double   velx;          //   8 bytes - velocity x
    double   vely;          //   8 bytes - velocity y
};                          // Total: 192 bytes
#pragma pack(pop)

class SharedMemoryPublisher
{
public:
    SharedMemoryPublisher(
        const std::string& frame_shm_name,
        int frame_w, int frame_h,
        int frame_slots,
        const std::string& tracking_shm_name,
        int tracking_slots
    );
    ~SharedMemoryPublisher();

    /// Publish a grayscale remapped frame to the frame ring buffer.
    void writeFrame(const cv::Mat& remap_grey);

    /// Publish tracking data to the tracking ring buffer.
    void writeTrackingData(const TrackingSlot& slot);

    /// Also publish the full grey camera frame (src_w x src_h) to a third
    /// region, every `every`-th frame (every <= 0: off).  Layout: uint64 write
    /// counter, uint32 width, uint32 height, then `slots` frames.  Returns
    /// false (raw output off, the other regions unaffected) if it cannot be
    /// created.
    bool enableRaw(const std::string& raw_shm_name, int src_w, int src_h,
                   int slots, int every);

    /// Call once per grabbed frame; copies only every `every`-th one.
    void writeRaw(const cv::Mat& frame_grey);

    bool isActive() const { return _active; }

    int frameW() const { return _frame_w; }
    int frameH() const { return _frame_h; }

private:
    static const size_t HEADER_SIZE = 8;
    static const size_t RAW_HEADER_SIZE = 16;

    // Raw (full camera frame) shared memory
    void*    _raw_shm_handle = nullptr;
    uint8_t* _raw_shm_ptr = nullptr;
    size_t   _raw_shm_size = 0;
    int      _raw_w = 0, _raw_h = 0;
    int      _raw_slots = 0;
    size_t   _raw_bytes = 0;
    int      _raw_every = 0;
    uint64_t _raw_seen = 0;

    // Frame shared memory
    void*    _frame_shm_handle;
    uint8_t* _frame_shm_ptr;
    size_t   _frame_shm_size;
    int      _frame_w, _frame_h;
    int      _frame_slots;
    size_t   _frame_bytes;

    // Tracking shared memory
    void*    _tracking_shm_handle;
    uint8_t* _tracking_shm_ptr;
    size_t   _tracking_shm_size;
    int      _tracking_slots;

    bool _active;
};
