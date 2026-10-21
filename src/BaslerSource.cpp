/// FicTrac http://rjdmoore.net/fictrac/
/// \file       BaslerSource.cpp
/// \brief      Basler USB3 sources (Pylon SDK).
/// \author     Wenbin Yang, Richard Moore
/// \copyright  CC BY-NC-SA 3.0

#if defined(BASLER_USB3)

#include "BaslerSource.h"

#include "Logger.h"
#include "timing.h"

#include <algorithm>

using namespace std;
using namespace cv;
using namespace Pylon;

namespace {

/// Value of the first readable float node in names, or -1 if none is.
/// USB3 (SFNC) cameras name the frame-rate nodes without the "Abs" suffix
/// that older GigE cameras use, so both are tried.
double readFloat(GenApi::INodeMap& nm, std::initializer_list<const char*> names)
{
    for (const char* name : names) {
        GenApi::CFloatPtr node = nm.GetNode(name);
        if (GenApi::IsReadable(node)) { return node->GetValue(); }
    }
    return -1;
}

} // namespace

BaslerSource::BaslerSource(int index)
{
    try {
        PylonInitialize();

        // Open the index-th enumerated device (same order as pypylon's
        // TlFactory.EnumerateDevices, which the acquisition code uses to map
        // the camera serial to src_fn).
        CTlFactory& tl = CTlFactory::GetInstance();
        DeviceInfoList_t devices;
        tl.EnumerateDevices(devices);
        if ((index < 0) || (index >= static_cast<int>(devices.size()))) {
            LOG_ERR("Error! Basler camera index %d out of range (%d camera(s) found).", index, static_cast<int>(devices.size()));
            return;
        }
        _cam.Attach(tl.CreateDevice(devices[index]));
        _cam.Open();

        LOG("Opening Basler camera device: %s (serial %s)", _cam.GetDeviceInfo().GetModelName().c_str(), _cam.GetDeviceInfo().GetSerialNumber().c_str());

        // Convert every frame to BGR8, the layout grab() hands to OpenCV.
        _converter.OutputPixelFormat = PixelType_BGR8packed;

        GenApi::INodeMap& control = _cam.GetNodeMap();
        const GenApi::CIntegerPtr camWidth = control.GetNode("Width");
        const GenApi::CIntegerPtr camHeight = control.GetNode("Height");
        _width = static_cast<int>(camWidth->GetValue());
        _height = static_cast<int>(camHeight->GetValue());

        // Start acquisition
        _cam.StartGrabbing();
        _fps = getFPS();

        LOG("Basler camera initialised (%dx%d @ %.3f fps)!", _width, _height, _fps);

        _open = true;
        _live = true;
    }
    catch (const GenericException &e) {
        LOG_ERR("Error opening capture device! Error was: %s", e.GetDescription());
    }
}

BaslerSource::~BaslerSource()
{
    try {
        if (_cam.IsGrabbing()) { _cam.StopGrabbing(); }
        if (_cam.IsOpen()) { _cam.Close(); }
        _cam.DestroyDevice();
    }
    catch (const GenericException &e) {
        LOG_ERR("Error closing capture device! Error was: %s", e.GetDescription());
    }
    _open = false;
    PylonTerminate();
}

double BaslerSource::getFPS()
{
    if (!_cam.IsOpen()) { return -1; }
    try {
        // The rate the camera actually delivers (exposure and ROI permitting),
        // then the requested rate if the camera has no resulting-rate node.
        return readFloat(_cam.GetNodeMap(), { "ResultingFrameRate", "ResultingFrameRateAbs", "AcquisitionFrameRate", "AcquisitionFrameRateAbs" });
    }
    catch (const GenericException &e) {
        LOG_ERR("Error reading frame rate! Error was: %s", e.GetDescription());
        return -1;
    }
}

bool BaslerSource::setFPS(double fps)
{
    bool ret = false;
    if (_cam.IsOpen() && (fps > 0)) {
        try {
            GenApi::INodeMap& control = _cam.GetNodeMap();
            GenApi::CBooleanPtr enable = control.GetNode("AcquisitionFrameRateEnable");
            if (GenApi::IsWritable(enable)) { enable->SetValue(true); }
            for (const char* name : { "AcquisitionFrameRate", "AcquisitionFrameRateAbs" }) {
                GenApi::CFloatPtr rate = control.GetNode(name);
                if (GenApi::IsWritable(rate)) {
                    rate->SetValue(std::min(std::max(fps, rate->GetMin()), rate->GetMax()));
                    ret = true;
                    break;
                }
            }
            if (!ret) {
                LOG_ERR("Error setting frame rate!");
            }
        }
        catch (const GenericException &e) {
            LOG_ERR("Error setting frame rate! Error was: %s", e.GetDescription());
        }
        _fps = getFPS();
        LOG("Device frame rate is now %.2f", _fps);
    }
    return ret;
}

bool BaslerSource::grab(cv::Mat& frame)
{
    if (!_open) { return false; }

    // Set grab timeout
    long int timeout = _fps > 0 ? max(static_cast<long int>(1000), static_cast<long int>(1000. / _fps)) : 1000; // set capture timeout to at least 1000 ms
    try {
        _cam.RetrieveResult(timeout, _ptrGrabResult, TimeoutHandling_ThrowException);
        double ts = ts_ms();    // backup, in case the device timestamp is junk
        _ms_since_midnight = ms_since_midnight();
        if (!_ptrGrabResult->GrabSucceeded()) {
            LOG_ERR("Error! Image capture failed (%d: %s).", _ptrGrabResult->GetErrorCode(), _ptrGrabResult->GetErrorDescription().c_str());
            // release the original image pointer
            _ptrGrabResult.Release();
            return false;
        }
        // Device tick count (ns on USB3 ace 2), as PGRSource reports the
        // Spinnaker device timestamp.
        _timestamp = static_cast<double>(_ptrGrabResult->GetTimeStamp());
        LOG_DBG("Frame captured %dx%d @ %f (t_sys: %f ms, t_day: %f ms)", _ptrGrabResult->GetWidth(), _ptrGrabResult->GetHeight(), _timestamp, ts, _ms_since_midnight);
        if (_timestamp <= 0) {
            _timestamp = ts;
        }
    }
    catch (const GenericException &e) {
        LOG_ERR("Error grabbing frame! Error was: %s", e.GetDescription());
        // release the original image pointer
        _ptrGrabResult.Release();
        return false;
    }

    try {
        // Convert image (mono or Bayer) to BGR8
        _converter.Convert(_pylonImg, _ptrGrabResult);

        Mat tmp(static_cast<int>(_pylonImg.GetHeight()), static_cast<int>(_pylonImg.GetWidth()), CV_8UC3, (uint8_t*)_pylonImg.GetBuffer());
        tmp.copyTo(frame);

        // release the original image pointer
        _ptrGrabResult.Release();
    }
    catch (const GenericException &e) {
        LOG_ERR("Error converting frame! Error was: %s", e.GetDescription());
        // release the original image pointer
        _ptrGrabResult.Release();
        return false;
    }

    return true;
}

#endif // BASLER_USB3
