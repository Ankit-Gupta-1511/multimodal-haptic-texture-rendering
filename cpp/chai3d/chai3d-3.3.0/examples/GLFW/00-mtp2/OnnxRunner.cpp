#include "OnnxRunner.h"
#include <fstream>
#include <iostream>
#include <cstring>
#include <algorithm>

#ifdef HAS_ORT
#  include <onnxruntime_cxx_api.h>
#endif

namespace {
    static std::string FindAssetUp(const std::string& fname) {
        // Try a few relative locations upward
        const char* bases[] = {"assets/", "../assets/", "../../assets/", "../../../assets/", "../../../../assets/"};
        for (auto b : bases) {
            std::string p = std::string(b) + fname;
            std::ifstream f(p, std::ios::binary);
            if (f.good()) return p;
        }
        return std::string();
    }

    static bool FileExists(const std::string& p) {
        std::ifstream f(p, std::ios::binary);
        return f.good();
    }

    struct ModelMeta { bool use_prev=false; bool has_audio=false; int fs_vib=1000; int fs_aud=8000; };

    static ModelMeta ParseModelJson(const std::string& path) {
        ModelMeta m{};
        std::ifstream f(path);
        if (!f.good()) return m;
        std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        auto findTrue = [&](const char* key){ auto pos = s.find(std::string("\"")+key+"\""); if (pos==std::string::npos) return false; auto t = s.find("true", pos); return t!=std::string::npos; };
        m.use_prev  = findTrue("use_prev");
        m.has_audio = (s.find("\"multitask_av\"") != std::string::npos) || (s.find("\"audio\"") != std::string::npos);
        auto findInt = [&](const char* key, int def){ auto pos = s.find(std::string("\"")+key+"\""); if (pos==std::string::npos) return def; pos = s.find(':', pos); if (pos==std::string::npos) return def; size_t e=pos+1; while (e<s.size() && (s[e]==' '||s[e]=='\t')) ++e; int val=def; try { val = std::stoi(s.substr(e)); } catch(...){} return val; };
        m.fs_vib = findInt("fs_vib", 1000);
        m.fs_aud = findInt("fs_aud", 8000);
        return m;
    }
}

struct OnnxRunner::Impl {
    bool ready=false;
    bool use_prev=false;
    bool has_audio=false;
    int fs_vib=1000;
    int fs_aud=8000;
    std::string model_path;
#ifdef HAS_ORT
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "mtp2"};
    Ort::SessionOptions opts;
    std::unique_ptr<Ort::Session> session;
#endif
};

OnnxRunner::OnnxRunner(const std::string& modelPath) : impl_(new Impl()) {
    std::string mpath = modelPath;
    if (mpath.empty()) {
        mpath = FindAssetUp("model.onnx");
    }
    impl_->model_path = mpath;
    std::string jpath = FindAssetUp("model.json");
    ModelMeta meta = ParseModelJson(jpath);
    impl_->use_prev  = meta.use_prev;
    impl_->has_audio = meta.has_audio;
    impl_->fs_vib    = meta.fs_vib;
    impl_->fs_aud    = meta.fs_aud;

#ifdef HAS_ORT
    if (!mpath.empty() && FileExists(mpath)) {
        try {
            impl_->opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
            // ORT uses wide strings for paths on Windows
            #if defined(_WIN32) || defined(_WIN64)
            std::wstring wpath(mpath.begin(), mpath.end());
            impl_->session = std::make_unique<Ort::Session>(impl_->env, wpath.c_str(), impl_->opts);
            #else
            impl_->session = std::make_unique<Ort::Session>(impl_->env, mpath.c_str(), impl_->opts);
            #endif
            impl_->ready = true;
        } catch (const std::exception& e) {
            std::cerr << "OnnxRunner: failed to create session: " << e.what() << std::endl;
            impl_->ready = false;
        }
    } else {
        std::cerr << "OnnxRunner: model not found (searched " << mpath << ")" << std::endl;
        impl_->ready = false;
    }
#else
    (void)mpath; // unused
    impl_->ready = false;
#endif
}

OnnxRunner::~OnnxRunner() { delete impl_; }

bool OnnxRunner::ready() const { return impl_->ready; }
bool OnnxRunner::usePrev() const { return impl_->use_prev; }
bool OnnxRunner::hasAudio() const { return impl_->has_audio; }
int  OnnxRunner::fsVib() const { return impl_->fs_vib; }
int  OnnxRunner::fsAud() const { return impl_->fs_aud; }

bool OnnxRunner::runDummy(std::vector<int64_t>& vibShape, std::vector<int64_t>& audShape) {
    vibShape.clear(); audShape.clear();
#ifdef HAS_ORT
    if (!impl_->ready) return false;
    try {
        // Build dummy inputs (B=1)
        const int64_t B=1;
        std::vector<int64_t> shpPatch{B,1,96,96};
        std::vector<int64_t> shpState{B,2};
        std::vector<float> patch(1*1*96*96, 0.0f);
        std::vector<float> state(2, 0.5f);
        std::vector<const char*> namesIn;
        std::vector<Ort::Value> valsIn;
        Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        namesIn.push_back("patch");
        valsIn.emplace_back(Ort::Value::CreateTensor<float>(mem, patch.data(), patch.size(), shpPatch.data(), shpPatch.size()));
        namesIn.push_back("state");
        valsIn.emplace_back(Ort::Value::CreateTensor<float>(mem, state.data(), state.size(), shpState.data(), shpState.size()));
        std::vector<float> prev; std::vector<int64_t> shpPrev;
        if (impl_->use_prev) {
            shpPrev = {B,100}; prev.assign(100, 0.0f);
            namesIn.push_back("prev_vib");
            valsIn.emplace_back(Ort::Value::CreateTensor<float>(mem, prev.data(), prev.size(), shpPrev.data(), shpPrev.size()));
        }

        // Execute
        std::vector<const char*> outNames;
        if (impl_->has_audio) { outNames = {"vib", "audio"}; } else { outNames = {"vib"}; }
        auto outs = impl_->session->Run(Ort::RunOptions{nullptr}, namesIn.data(), valsIn.data(), (size_t)namesIn.size(), outNames.data(), (size_t)outNames.size());
        // Read shapes
        auto& vib = outs[0];
        Ort::TensorTypeAndShapeInfo vibInfo = vib.GetTensorTypeAndShapeInfo();
        vibShape = vibInfo.GetShape();
        if (impl_->has_audio && outs.size() > 1) {
            auto& aud = outs[1];
            Ort::TensorTypeAndShapeInfo audInfo = aud.GetTensorTypeAndShapeInfo();
            audShape = audInfo.GetShape();
        }
        return true;
    } catch (const std::exception& e) {
        std::cerr << "OnnxRunner: runDummy failed: " << e.what() << std::endl;
        return false;
    }
#else
    (void)vibShape; (void)audShape; return false;
#endif
}

void OnnxRunner::SelfTest() {
    std::cout << "[OnnxRunner] SelfTest: "
#ifdef HAS_ORT
              << "HAS_ORT, "
#else
              << "NO_ORT, "
#endif
              << "attempting to load assets/model.onnx" << std::endl;
    OnnxRunner R;
    if (!R.ready()) { std::cout << "[OnnxRunner] Not ready (missing ORT or model)." << std::endl; return; }
    std::vector<int64_t> vs, as;
    if (R.runDummy(vs, as)) {
        auto printShape = [](const std::vector<int64_t>& sh){ std::cout << "("; for (size_t i=0;i<sh.size();++i){ std::cout<<sh[i]; if(i+1<sh.size()) std::cout<<","; } std::cout << ")"; };
        std::cout << "[OnnxRunner] vib shape="; printShape(vs);
        if (!as.empty()) { std::cout << ", audio shape="; printShape(as); }
        std::cout << ", use_prev=" << (R.usePrev()?"true":"false") << ", fs_vib=" << R.fsVib() << ", fs_aud=" << R.fsAud() << std::endl;
    } else {
        std::cout << "[OnnxRunner] runDummy failed." << std::endl;
    }
}
