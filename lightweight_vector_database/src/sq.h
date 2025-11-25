#ifndef SQ_H
#define SQ_H

#include <vector>
#include <cstdint>
#include <algorithm>
#include <fstream>

namespace sq {

class ScalarQuantizer {
public:
    ScalarQuantizer(size_t original_dim)
        : original_dim_(original_dim) {}

    void train(const std::vector<std::vector<float>>& training_data) {
        if (training_data.empty()) {
            return;
        }
        mins_.resize(original_dim_);
        maxs_.resize(original_dim_);
        for (size_t i = 0; i < original_dim_; ++i) {
            mins_[i] = training_data[0][i];
            maxs_[i] = training_data[0][i];
        }
        for (const auto& vec : training_data) {
            for (size_t i = 0; i < original_dim_; ++i) {
                if (vec[i] < mins_[i]) {
                    mins_[i] = vec[i];
                }
                if (vec[i] > maxs_[i]) {
                    maxs_[i] = vec[i];
                }
            }
        }
    }

    std::vector<uint8_t> encode(const std::vector<float>& vector) const {
        if (mins_.empty() || maxs_.empty()) {
            throw std::runtime_error("Quantizer is not trained.");
        }
        std::vector<uint8_t> encoded(original_dim_);
        for (size_t i = 0; i < original_dim_; ++i) {
            float range = maxs_[i] - mins_[i];
            if (range == 0) {
                encoded[i] = 0;
            } else {
                float scaled = (vector[i] - mins_[i]) / range;
                encoded[i] = static_cast<uint8_t>(std::round(scaled * 255.0f));
            }
        }
        return encoded;
    }

    std::vector<float> decode(const std::vector<uint8_t>& vector) const {
        if (mins_.empty() || maxs_.empty()) {
            throw std::runtime_error("Quantizer is not trained.");
        }
        std::vector<float> decoded(original_dim_);
        for (size_t i = 0; i < original_dim_; ++i) {
            float range = maxs_[i] - mins_[i];
            if (range == 0) {
                decoded[i] = mins_[i];
            } else {
                decoded[i] = mins_[i] + (vector[i] / 255.0f) * range;
            }
        }
        return decoded;
    }

    float calculate_distance(const std::vector<float>& query_vector, const std::vector<uint8_t>& encoded_vector) const {
        std::vector<float> decoded_vector = decode(encoded_vector);
        float distance = 0.0f;
        for (size_t i = 0; i < original_dim_; ++i) {
            float diff = query_vector[i] - decoded_vector[i];
            distance += diff * diff;
        }
        return distance;
    }

    void save(std::ofstream& ofs) const {
        ofs.write(reinterpret_cast<const char*>(&original_dim_), sizeof(original_dim_));
        ofs.write(reinterpret_cast<const char*>(mins_.data()), original_dim_ * sizeof(float));
        ofs.write(reinterpret_cast<const char*>(maxs_.data()), original_dim_ * sizeof(float));
    }

    void load(std::ifstream& ifs) {
        ifs.read(reinterpret_cast<char*>(&original_dim_), sizeof(original_dim_));
        mins_.resize(original_dim_);
        maxs_.resize(original_dim_);
        ifs.read(reinterpret_cast<char*>(mins_.data()), original_dim_ * sizeof(float));
        ifs.read(reinterpret_cast<char*>(maxs_.data()), original_dim_ * sizeof(float));
    }

    bool is_trained() const {
        return !mins_.empty();
    }

    size_t get_original_dim() const {
        return original_dim_;
    }

private:
    size_t original_dim_;
    std::vector<float> mins_;
    std::vector<float> maxs_;
};

} // namespace sq

#endif // SQ_H
