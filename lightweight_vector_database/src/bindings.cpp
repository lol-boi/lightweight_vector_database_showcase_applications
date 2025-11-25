#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/chrono.h>

#include "database.h"
#include "hnsw.h"
#include "sq.h"

namespace py = pybind11;
using namespace hnsw;
using namespace sq;

PYBIND11_MODULE(vector_database_bindings, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    py::enum_<DistanceMetric>(m, "DistanceMetric")
        .value("L2", DistanceMetric::L2)
        .value("COSINE", DistanceMetric::COSINE)
        .value("IP", DistanceMetric::IP)
        .export_values();

    py::enum_<Include>(m, "Include")
        .value("ID", Include::ID)
        .value("DISTANCE", Include::DISTANCE)
        .value("METADATA", Include::METADATA)
        .value("VECTOR", Include::VECTOR)
        .export_values();

    py::enum_<SyncMode>(m, "SyncMode")
        .value("FULL", SyncMode::FULL)
        .value("NORMAL", SyncMode::NORMAL)
        .value("OFF", SyncMode::OFF)
        .export_values();

    py::class_<QueryResult>(m, "QueryResult")
        .def(py::init<>())
        .def_readwrite("id", &QueryResult::id)
        .def_readwrite("distance", &QueryResult::distance)
        .def_readwrite("metadata", &QueryResult::metadata)
        .def_readwrite("vector", &QueryResult::vector);

    py::class_<ScalarQuantizer>(m, "ScalarQuantizer")
        .def(py::init<size_t>())
        .def("train", &ScalarQuantizer::train)
        .def("encode", &ScalarQuantizer::encode)
        .def("decode", &ScalarQuantizer::decode)
        .def("calculate_distance", &ScalarQuantizer::calculate_distance)
        .def("is_trained", &ScalarQuantizer::is_trained)
        .def("get_original_dim", &ScalarQuantizer::get_original_dim);

    py::class_<Database>(m, "Database")
        .def(py::init<const std::string&, size_t, int, int, int, DistanceMetric, bool, size_t, bool>(),
             py::arg("db_path"), py::arg("vector_dimension"), py::arg("M") = 16,
             py::arg("efConstruction") = 200, py::arg("efSearch") = 50,
             py::arg("metric") = DistanceMetric::L2, py::arg("read_only") = false,
             py::arg("cache_size_mb") = 0, py::arg("sq_enabled") = false)
        .def("insert", &Database::insert, py::arg("vec"), py::arg("meta") = Metadata{})
        .def("update_vector", &Database::update_vector, py::arg("id"), py::arg("new_vec"), py::arg("new_meta") = Metadata{})
        .def("delete_vector", &Database::delete_vector, py::arg("id"))
        .def("query", &Database::query, py::arg("query"), py::arg("k"),
             py::arg("filter") = nullptr, py::arg("include") = std::set<Include>{Include::ID})
        .def("train_quantizer", &Database::train_quantizer)
        .def("rebuild_index", &Database::rebuild_index)
        .def("save", &Database::save, py::arg("sync_mode") = SyncMode::FULL)
        .def("load", &Database::load);
}
