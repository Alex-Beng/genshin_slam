#include "slam/capture.h"

namespace slam {

Capture::Capture() : source_(Source::Video) {}
Capture::~Capture() = default;

bool Capture::postprocess(cv::Mat& frame) {
    if (frame.empty()) return false;
    if (!target_.empty() && (target_.width > 0 && target_.height > 0) &&
        (frame.cols != target_.width || frame.rows != target_.height)) {
        cv::resize(frame, frame, target_, 0, 0, cv::INTER_AREA);
    }
    size_ = cv::Size(frame.cols, frame.rows);
    return true;
}

// ---------------------------------------------------------------------------
// VideoFileCapture
// ---------------------------------------------------------------------------
VideoFileCapture::VideoFileCapture(const std::string& path) : path_(path) {
    source_ = Source::Video;
}

bool VideoFileCapture::init() {
    cap_.open(path_);
    if (!cap_.isOpened()) return false;
    fps_ = cap_.get(cv::CAP_PROP_FPS);
    frame_count_ = (int)cap_.get(cv::CAP_PROP_FRAME_COUNT);
    return true;
}

void VideoFileCapture::release() {
    cap_.release();
    frame_count_ = 0;
    fps_ = 0.0;
}

bool VideoFileCapture::read(cv::Mat& frame) {
    if (!cap_.isOpened()) return false;
    if (!cap_.read(frame)) return false;
    return postprocess(frame);
}

#ifdef _WIN32
// ---------------------------------------------------------------------------
// BitbltCapture
// ---------------------------------------------------------------------------
BitbltCapture::BitbltCapture(const std::wstring& title, int max_w, int max_h)
    : title_(title), max_w_(max_w), max_h_(max_h) {
    source_ = Source::Bitblt;
}

bool BitbltCapture::findWindow() {
    hwnd_ = FindWindowW(nullptr, title_.c_str());
    return hwnd_ != nullptr;
}

bool BitbltCapture::init() {
    if (!findWindow()) return false;
    return true;
}

void BitbltCapture::release() {
    if (bmp_) { DeleteObject(bmp_); bmp_ = nullptr; }
    if (mem_dc_) { DeleteDC(mem_dc_); mem_dc_ = nullptr; }
    hwnd_ = nullptr;
}

bool BitbltCapture::read(cv::Mat& frame) {
    if (!hwnd_ || !IsWindow(hwnd_)) {
        if (!findWindow()) return false;
    }

    RECT rc;
    if (!GetClientRect(hwnd_, &rc)) return false;
    int cw = rc.right - rc.left;
    int ch = rc.bottom - rc.top;
    if (cw <= 0 || ch <= 0) return false;

    // DPI awareness for the monitor where the window is.
    HMONITOR mon = MonitorFromWindow(hwnd_, MONITOR_DEFAULTTONEAREST);
    MONITORINFOEXW mi;
    mi.cbSize = sizeof(mi);
    GetMonitorInfoW(mon, &mi);
    DEVMODEW dm;
    dm.dmSize = sizeof(dm);
    dm.dmDriverExtra = 0;
    EnumDisplaySettingsW(mi.szDevice, ENUM_CURRENT_SETTINGS, &dm);
    double scale_x = (double)dm.dmPelsWidth / (double)(mi.rcMonitor.right - mi.rcMonitor.left);
    double scale_y = (double)dm.dmPelsHeight / (double)(mi.rcMonitor.bottom - mi.rcMonitor.top);

    HDC hdc = GetDC(hwnd_);
    if (!hdc) return false;
    if (mem_dc_) DeleteDC(mem_dc_);
    mem_dc_ = CreateCompatibleDC(hdc);

    int pw = (int)std::lround(cw * scale_x);
    int ph = (int)std::lround(ch * scale_y);
    if (pw <= 0 || ph <= 0) { ReleaseDC(hwnd_, hdc); return false; }

    if (bmp_) DeleteObject(bmp_);
    bmp_ = CreateCompatibleBitmap(hdc, cw, ch);
    HGDIOBJ old = SelectObject(mem_dc_, bmp_);

    // BitBlt in physical pixels but bitmap in logical; select the window DC
    // directly ensures the capture matches what the window draws.
    RECT wrc;
    GetWindowRect(hwnd_, &wrc);
    BitBlt(mem_dc_, 0, 0, cw, ch, GetDC(GetDesktopWindow()),
           wrc.left / 1, wrc.top / 1, SRCCOPY);

    SelectObject(mem_dc_, old);
    ReleaseDC(hwnd_, hdc);

    BITMAP bm;
    GetObject(bmp_, sizeof(BITMAP), &bm);
    int nCh = bm.bmBitsPixel / 8;
    frame = cv::Mat(bm.bmHeight, bm.bmWidth, CV_MAKETYPE(CV_8U, nCh == 3 ? 3 : 4));
    GetBitmapBits(bmp_, bm.bmHeight * bm.bmWidth * nCh, frame.data);

    if (max_w_ > 0 && max_h_ > 0) {
        double s = std::min((double)max_w_ / frame.cols, (double)max_h_ / frame.rows);
        if (s < 1.0) cv::resize(frame, frame, cv::Size(), s, s, cv::INTER_AREA);
    }
    return postprocess(frame);
}
#endif // _WIN32

} // namespace slam