#ifndef HNSW_H
#define HNSW_H

#include <vector>
#include <cstdint>
#include <cmath>
#include <stdexcept>
#include <queue>
#include <set>
#include <random>
#include <algorithm>
#include <numeric>
#include <map>
#include <functional>
#include <unordered_set>
#include "sq.h"

namespace hnsw {

// Enum for supported distance metrics
enum class DistanceMetric {
    L2,
    COSINE,
    IP // Inner Product
};

// Type alias for metadata
using Metadata = std::map<std::string, std::string>;

// Type alias for a filter function
using FilterFunc = std::function<bool(const Metadata&)>;

// Enum to control what data is returned in the search result
enum class Include {
    ID,
    DISTANCE,
    METADATA,
    VECTOR
};

// Struct for search results
struct QueryResult {
    int id;
    float distance;
    Metadata metadata;
    std::vector<float> vector;
};

// Represents a node in the HNSW graph.
struct Node {
    uint32_t id;
    int max_layer;
    std::vector<std::vector<int>> neighbors;

    Node(uint32_t id, int max_layer) : id(id), max_layer(max_layer) {
        neighbors.resize(max_layer + 1);
    }
};

class VectorStorage {
public:
    VectorStorage(size_t vector_dimension, sq::ScalarQuantizer* sq = nullptr) 
        : vector_dimension_(vector_dimension), sq_(sq) {}

    void add_vector(const std::vector<float>& vec, const Metadata& meta) {
        if (vec.size() != vector_dimension_) {
            throw std::invalid_argument("Vector dimension mismatch.");
        }
        vectors_.push_back(vec);
        metadata_.push_back(meta);
        if (sq_ && sq_->is_trained()) {
            encoded_vectors_.push_back(sq_->encode(vec));
        }
    }

    void encode_all_vectors() {
        if (!sq_ || !sq_->is_trained()) {
            return;
        }
        encoded_vectors_.resize(vectors_.size());
        for (size_t i = 0; i < vectors_.size(); ++i) {
            encoded_vectors_[i] = sq_->encode(vectors_[i]);
        }
    }

    const std::vector<float>& get_vector(size_t index) const {
        return vectors_[index];
    }

    const std::vector<uint8_t>& get_encoded_vector(size_t index) const {
        if (!sq_) {
            throw std::runtime_error("Quantizer is not enabled.");
        }
        return encoded_vectors_[index];
    }

    const Metadata& get_metadata(size_t index) const {
        return metadata_[index];
    }

    size_t size() const {
        return vectors_.size();
    }

    size_t get_vector_dimension() const {
        return vector_dimension_;
    }

private:
    size_t vector_dimension_;
    std::vector<std::vector<float>> vectors_;
    std::vector<Metadata> metadata_;
    sq::ScalarQuantizer* sq_ = nullptr;
    std::vector<std::vector<uint8_t>> encoded_vectors_;
};

class HNSW {
public:
    HNSW(size_t vector_dimension, int M = 5, int efConstruction = 10, int efSearch = 10, DistanceMetric metric = DistanceMetric::L2, sq::ScalarQuantizer* sq = nullptr)
        : vector_storage(vector_dimension, sq),
          entry_point_id(-1),
          M(M),
          efConstruction(efConstruction),
          efSearch(efSearch),
          distance_metric(metric),
          sq_(sq),
          gen(std::random_device{}()),
          dist(0.0, 1.0) {
        m_L = 1.0 / log(1.0 * M);
    }

    HNSW(size_t vector_dimension, int M, int efConstruction, int efSearch, DistanceMetric metric, const std::vector<Node>& nodes, const VectorStorage& vector_storage, const std::unordered_set<uint32_t>& deleted_nodes, sq::ScalarQuantizer* sq = nullptr)
        : vector_storage(vector_storage),
          nodes(nodes),
          deleted_nodes_(deleted_nodes),
          entry_point_id(nodes.empty() ? -1 : nodes.back().id),
          M(M),
          efConstruction(efConstruction),
          efSearch(efSearch),
          distance_metric(metric),
          sq_(sq),
          gen(std::random_device{}()),
          dist(0.0, 1.0) {
        m_L = 1.0 / log(1.0 * M);
    }

    template<typename T>
    void set_quantizer(T* quantizer) {
        if constexpr (std::is_same_v<T, sq::ScalarQuantizer>) {
            sq_ = quantizer;
            vector_storage = VectorStorage(vector_storage.get_vector_dimension(), sq_);
        }
    }
    
    std::vector<int> search_layer(const std::vector<float>& query, int entry_point_id, int ef, int layer_level, const FilterFunc& filter = nullptr) {
        using Candidate = std::pair<float, int>;
        std::priority_queue<Candidate, std::vector<Candidate>, std::greater<Candidate>> candidate_queue;
        std::priority_queue<Candidate> result_queue;
        std::set<int> visited_nodes;

        float dist_to_entry = calculate_distance(query, entry_point_id);
        if (deleted_nodes_.find(entry_point_id) == deleted_nodes_.end()) {
            candidate_queue.push({dist_to_entry, entry_point_id});
            if (!filter || filter(vector_storage.get_metadata(entry_point_id))) {
                result_queue.push({dist_to_entry, entry_point_id});
            }
        }
        visited_nodes.insert(entry_point_id);

        while (!candidate_queue.empty()) {
            Candidate current = candidate_queue.top();
            candidate_queue.pop();

            if (result_queue.size() == ef && current.first > result_queue.top().first) {
                break;
            }

            for (int neighbor_id : nodes[current.second].neighbors[layer_level]) {
                if (visited_nodes.find(neighbor_id) == visited_nodes.end()) {
                    visited_nodes.insert(neighbor_id);

                    if (deleted_nodes_.find(neighbor_id) != deleted_nodes_.end()) {
                        continue;
                    }

                    float dist_to_neighbor = calculate_distance(query, neighbor_id);
                    if (result_queue.size() < ef || dist_to_neighbor < result_queue.top().first) {
                        candidate_queue.push({dist_to_neighbor, neighbor_id});
                        if (!filter || filter(vector_storage.get_metadata(neighbor_id))) {
                            result_queue.push({dist_to_neighbor, neighbor_id});
                        }

                        while (result_queue.size() > ef) {
                            result_queue.pop();
                        }
                    }
                }
            }
        }

        std::vector<int> final_results;
        while (!result_queue.empty()) {
            final_results.push_back(result_queue.top().second);
            result_queue.pop();
        }
        std::reverse(final_results.begin(), final_results.end());
        return final_results;
    }

    uint32_t insert(const std::vector<float>& vec, const Metadata& meta = {}) {
        uint32_t new_node_id = vector_storage.size();
        vector_storage.add_vector(vec, meta);

        int new_node_layer = random_level();
        nodes.emplace_back(new_node_id, new_node_layer);

        if (entry_point_id == -1) {
            entry_point_id = new_node_id;
            return new_node_id;
        }

        int current_node_id = entry_point_id;
        int current_max_layer = nodes[current_node_id].max_layer;

        for (int layer = current_max_layer; layer > new_node_layer; --layer) {
            std::vector<int> candidates = search_layer(vec, current_node_id, 1, layer);
            if (candidates.empty()) break;
            current_node_id = candidates[0];
        }

        for (int layer = std::min(new_node_layer, current_max_layer); layer >= 0; --layer) {
            std::vector<int> neighbors_found = search_layer(vec, current_node_id, efConstruction, layer);
            if (neighbors_found.empty()) continue;

            std::vector<int> new_node_neighbors;
            for (int neighbor_id : neighbors_found) {
                if (new_node_neighbors.size() < M) {
                    new_node_neighbors.push_back(neighbor_id);
                } else break;
            }

            for (int neighbor_id : new_node_neighbors) {
                nodes[new_node_id].neighbors[layer].push_back(neighbor_id);
                nodes[neighbor_id].neighbors[layer].push_back(new_node_id);

                if (nodes[neighbor_id].neighbors[layer].size() > M) {
                    float max_dist = -1.0f;
                    int furthest_neighbor_idx = -1;
                    for (size_t i = 0; i < nodes[neighbor_id].neighbors[layer].size(); ++i) {
                        int current_connected_neighbor_id = nodes[neighbor_id].neighbors[layer][i];
                        float dist = calculate_distance(vector_storage.get_vector(neighbor_id), current_connected_neighbor_id);
                        if (dist > max_dist) {
                            max_dist = dist;
                            furthest_neighbor_idx = i;
                        }
                    }
                    if (furthest_neighbor_idx != -1) {
                        nodes[neighbor_id].neighbors[layer].erase(nodes[neighbor_id].neighbors[layer].begin() + furthest_neighbor_idx);
                    }
                }
            }
            current_node_id = neighbors_found[0];
        }

        if (new_node_layer > nodes[entry_point_id].max_layer) {
            entry_point_id = new_node_id;
        }
        return new_node_id;
    }

    std::vector<QueryResult> k_nearest_neighbors(const std::vector<float>& query, int k, const FilterFunc& filter = nullptr, const std::set<Include>& include = {Include::ID}) {
        if (entry_point_id == -1) return {};

        int current_node_id = entry_point_id;
        int current_max_layer = nodes[current_node_id].max_layer;

        for (int layer = current_max_layer; layer > 0; --layer) {
            std::vector<int> candidates = search_layer(query, current_node_id, 1, layer, filter);
            if (!candidates.empty()) {
                current_node_id = candidates[0];
            }
        }

        std::vector<int> results_ids = search_layer(query, current_node_id, std::max(k, efSearch), 0, filter);

        std::vector<QueryResult> final_results;
        for (int id : results_ids) {
            if (deleted_nodes_.count(id)) continue;
            if (final_results.size() >= k) break;
            
            QueryResult result;
            result.id = id;
            if (include.count(Include::DISTANCE)) {
                result.distance = calculate_distance(query, id);
            }
            if (include.count(Include::METADATA)) {
                result.metadata = vector_storage.get_metadata(id);
            }
            if (include.count(Include::VECTOR)) {
                result.vector = vector_storage.get_vector(id);
            }
            final_results.push_back(result);
        }
        return final_results;
    }

    size_t size() const { return nodes.size(); }
    const std::vector<Node>& get_nodes() const { return nodes; }
    int get_entry_point() const { return entry_point_id; }
    int get_M() const { return M; }
    int get_efConstruction() const { return efConstruction; }
    int get_efSearch() const { return efSearch; }
    DistanceMetric get_distance_metric() const { return distance_metric; }
    const VectorStorage& get_vector_storage() const { return vector_storage; }
    const std::unordered_set<uint32_t>& get_deleted_nodes() const { return deleted_nodes_; }

    void mark_deleted(uint32_t id) {
        deleted_nodes_.insert(id);
        if (entry_point_id == id) {
            int new_entry_point = -1;
            int max_layer = -1;
            for (const auto& node : nodes) {
                if (!deleted_nodes_.count(node.id)) {
                    if (node.max_layer > max_layer) {
                        max_layer = node.max_layer;
                        new_entry_point = node.id;
                    }
                }
            }
            entry_point_id = new_entry_point;
        }
    }

private:
    VectorStorage vector_storage;
    std::vector<Node> nodes;
    int entry_point_id;
    int M;
    int efConstruction;
    int efSearch;
    DistanceMetric distance_metric;
    sq::ScalarQuantizer* sq_ = nullptr;
    double m_L;
    std::mt19937 gen;
    std::uniform_real_distribution<> dist;
    std::unordered_set<uint32_t> deleted_nodes_;

    int random_level() {
        return static_cast<int>(floor(-log(dist(gen)) * m_L));
    }

    float calculate_distance(const std::vector<float>& a, const std::vector<float>& b) const {
        switch (distance_metric) {
            case DistanceMetric::L2: return calculate_l2_distance(a, b);
            case DistanceMetric::COSINE: return calculate_cosine_distance(a, b);
            case DistanceMetric::IP: return calculate_inner_product_distance(a, b);
            default: throw std::runtime_error("Unknown distance metric.");
        }
    }

    float calculate_distance(const std::vector<float>& query, int node_id) const {
        if (sq_ && sq_->is_trained()) {
            return sq_->calculate_distance(query, vector_storage.get_encoded_vector(node_id));
        }
        return calculate_distance(query, vector_storage.get_vector(node_id));
    }

    float calculate_l2_distance(const std::vector<float>& a, const std::vector<float>& b) const {
        float distance = 0.0f;
        for (size_t i = 0; i < a.size(); ++i) {
            float diff = a[i] - b[i];
            distance += diff * diff;
        }
        return distance;
    }

    float calculate_cosine_distance(const std::vector<float>& a, const std::vector<float>& b) const {
        float dot_product = std::inner_product(a.begin(), a.end(), b.begin(), 0.0f);
        float norm_a = std::sqrt(std::inner_product(a.begin(), a.end(), a.begin(), 0.0f));
        float norm_b = std::sqrt(std::inner_product(b.begin(), b.end(), b.begin(), 0.0f));
        if (norm_a == 0.0f || norm_b == 0.0f) return 1.0f;
        return 1.0f - (dot_product / (norm_a * norm_b));
    }

    float calculate_inner_product_distance(const std::vector<float>& a, const std::vector<float>& b) const {
        return -std::inner_product(a.begin(), a.end(), b.begin(), 0.0f);
    }
};

} // namespace hnsw

#endif // HNSW_H
