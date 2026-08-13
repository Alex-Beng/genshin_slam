#ifndef SLAM_CAPTURE_H
#define SLAM_CAPTURE_H

#include <opencv2/opencv.hpp>
#include <string>

#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#  include <cmath>
#endif

namespace slam {

// Capture abstraction layer: provides game frames from a video file or a
// live window (BitBlt). Runtime-switchable between sources.
class Capture {
public:
    enum class Source { Video, Bitblt };

    Capture();
    virtual ~Capture();

    Source source() const { return source_; }

    // Initialize the capture source. Returns false on failure.
    virtual bool init() = 0;
    virtual void release() = 0;

    // Grab the next frame. Returns false at end-of-stream / on failure.
    virtual bool read(cv::Mat& frame) = 0;

    // Current frame size (after resize to target_), or empty before first read.
    const cv::Size& size() const { return size_; }

    // Optional resize to a fixed size (empty = keep native).
    void setTargetSize(const cv::Size& s) { target_ = s; }

protected:
    Source source_;
    cv::Size size_;
    cv::Size target_;

    // Apply target resize if set.
    bool postprocess(cv::Mat& frame);
};

// Reads frames from a video file (cv::VideoCapture).
class VideoFileCapture : public Capture {
public:
    explicit VideoFileCapture(const std::string& path);
    bool init() override;
    void release() override;
    bool read(cv::Mat& frame) override;

    int frameCount() const { return frame_count_; }
    double fps() const { return fps_; }

private:
    std::string path_;
    cv::VideoCapture cap_;
    int frame_count_ = 0;
    double fps_ = 0.0;
};

#ifdef _WIN32
// Captures a game window via BitBlt (includes DPI-aware scaling).
// Window is matched by exact title, or the foreground window if title empty.
class BitbltCapture : public Capture {
public:
    explicit BitbltCapture(const std::wstring& title, int max_w = 0, int max_h = 0);
    bool init() override;
    void release() override;
    bool read(cv::Mat& frame) override;

private:
    std::wstring title_;
    HWND hwnd_ = nullptr;
    HBITMAP bmp_ = nullptr;
    HDC mem_dc_ = nullptr;
    int max_w_, max_h_;

    bool findWindow();
    bool captureImpl(cv::Mat& frame);
};
#endif // _WIN32

} // namespace slam

#endif // SLAM_CAPTURE_H