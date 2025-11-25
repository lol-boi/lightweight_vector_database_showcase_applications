# Lightweight Vector Database

A lightweight, header-only, in-memory vector database for C++17, built on the Hierarchical Navigable Small World (HNSW) algorithm for fast and efficient Approximate Nearest Neighbor (ANN) search.

## Features

-   **High-Speed ANN Search**: Utilizes the HNSW algorithm for state-of-the-art search performance.
-   **CRUD Operations**: Full support for Create, Read, Update, and Delete operations.
-   **Multiple Distance Metrics**: Supports L2 (Euclidean), Cosine, and Inner Product (IP) distances.
-   **Memory Optimization**: Includes optional **Scalar Quantization (SQ)** to significantly reduce memory footprint by compressing vectors to 8-bit integers.
-   **Rich Queries**: Supports metadata storage, `k`-NN search, and server-side filtering.
-   **Data Persistence**: Ability to save the database index and vectors to a single file and load it back into memory.
-   **Easy Integration**: Header-only library. Just include `src/database.h` in your project.
-   **Concurrency**: Designed for performance, though thread-safety for concurrent writes is not built-in and must be managed by the user.

## How to Build and Test

Since this is a header-only library, you only need to include the `src/database.h` file in your project. No separate compilation is required.

To build and run the provided tests, you can use CMake:

```bash
# Create a build directory
mkdir build && cd build

# Generate build files
cmake ..

# Compile the test executables
make

# Run the tests
ctest
```

## API Reference & Usage

The main entry point for all database operations is the `hnsw::Database` class.

### Initialization

To create or load a database, instantiate the `hnsw::Database` class.

```cpp
#include "src/database.h"

// Define database parameters
const std::string db_path = "my_database.bin";
const size_t vector_dimension = 128;

// Create a new database instance
hnsw::Database db(db_path, vector_dimension);
```

The constructor has several optional parameters to tune the HNSW algorithm and enable features:

`Database(db_path, vector_dimension, M, efConstruction, efSearch, metric, read_only, cache_size_mb, sq_enabled)`

-   `M`: The maximum number of neighbors per node in the graph (default: 16). Higher values can improve recall at the cost of memory and index build time.
-   `efConstruction`: The size of the dynamic candidate list during index construction (default: 200). Higher values lead to a better-quality index.
-   `efSearch`: The size of the dynamic candidate list during search (default: 50). Higher values improve recall at the cost of search speed.
-   `metric`: The distance metric to use. Can be `hnsw::DistanceMetric::L2` (default), `hnsw::DistanceMetric::COSINE`, or `hnsw::DistanceMetric::IP`.
-   `read_only`: Set to `true` to load an existing database in read-only mode.
-   `sq_enabled`: Set to `true` to enable Scalar Quantization.

### Inserting Data

Use the `insert` method to add vectors. You can also include a `Metadata` map. The method returns the unique ID of the new vector.

```cpp
#include <vector>
#include <map>

std::vector<float> my_vector(128, 0.5f); // Example vector
hnsw::Metadata my_metadata = {{"category", "A"}, {"timestamp", "2025-11-20"}};

uint32_t new_id = db.insert(my_vector, my_metadata);
// new_id will be 0, 1, 2, ...
```

### Querying Data

Use the `query` method to find the `k` nearest neighbors to a query vector.

```cpp
std::vector<float> query_vector(128, 0.55f);
int k = 5; // Find the 5 nearest neighbors

// Define what data to include in the results
std::set<hnsw::Include> include_data = {
    hnsw::Include::ID, 
    hnsw::Include::DISTANCE, 
    hnsw::Include::METADATA
};

// Perform the query
std::vector<hnsw::QueryResult> results = db.query(query_vector, k, nullptr, include_data);

for (const auto& result : results) {
    std::cout << "ID: " << result.id 
              << ", Distance: " << result.distance 
              << ", Category: " << result.metadata.at("category") 
              << std::endl;
}
```

### Filtering Queries

You can provide a lambda function to the `query` method to filter results based on their metadata *before* they are returned.

```cpp
// Filter for vectors where category is "A"
auto my_filter = [](const hnsw::Metadata& meta) {
    auto it = meta.find("category");
    return it != meta.end() && it->second == "A";
};

std::vector<hnsw::QueryResult> filtered_results = db.query(query_vector, k, my_filter);
```

### Update and Delete

This library uses a **soft-delete** mechanism for performance.

-   `delete_vector(id)`: Marks a vector as deleted. It will be excluded from all future queries but remains in the index.
-   `update_vector(id, new_vec, new_meta)`: Updates a vector by marking the old one as deleted and inserting the new one. **Note: This operation changes the vector's ID.**

```cpp
// Delete vector with ID 3
db.delete_vector(3);

// Update vector with ID 2
uint32_t updated_id = db.update_vector(2, new_vector, new_metadata);
std::cout << "Vector 2 was updated. Its new ID is: " << updated_id << std::endl;
```

### Index Rebuilding

Because deletions are "soft," the index can accumulate deleted nodes over time, which may slightly degrade search performance and use unnecessary memory. The `rebuild_index()` method creates a new, clean index containing only the active vectors.

You should call this method periodically if your application has a high volume of deletions or updates.

```cpp
// Rebuild the index to permanently remove all soft-deleted vectors
db.rebuild_index();
```

### Scalar Quantization (SQ)

SQ reduces memory usage by quantizing floating-point vectors into 8-bit integer vectors.

1.  **Enable it** in the constructor:
    ```cpp
    hnsw::Database db("sq_db.bin", 128, 16, 200, 50, hnsw::DistanceMetric::L2, false, 0, true);
    ```
2.  **Insert your data** as usual.
3.  **Call `rebuild_index()`**. This method will first train the quantizer on all the current data to find the optimal ranges, and then it will encode all vectors and build the HNSW index.

After this, all queries will use Asymmetric Distance Computation (ADC) for faster, memory-efficient searches.

### Persistence

You can save the entire database state (including the HNSW graph, vectors, and metadata) to disk and load it back.

```cpp
// Save the database to the path specified in the constructor
db.save();

// To load the database, create a new instance with the same path
hnsw::Database loaded_db("my_database.bin", 128, 16, 200, 50, hnsw::DistanceMetric::L2, true); // Use read_only mode

// The database is now ready to be queried
auto results = loaded_db.query(query_vector, 5);
```

## Complete Example: Getting Started

Here is a full example demonstrating the primary features:

```cpp
#include "src/database.h"
#include <iostream>
#include <vector>

int main() {
    const std::string db_path = "example_db.bin";
    const size_t vector_dimension = 2;

    // 1. Create a new database instance
    hnsw::Database db(db_path, vector_dimension);

    // 2. Insert vectors with metadata
    std::cout << "Inserting data..." << std::endl;
    db.insert({1.0f, 1.0f}, {{"type", "A"}}); // ID 0
    db.insert({1.1f, 1.2f}, {{"type", "A"}}); // ID 1
    db.insert({5.0f, 5.0f}, {{"type", "B"}}); // ID 2
    db.insert({5.2f, 5.1f}, {{"type", "B"}}); // ID 3

    // 3. Perform a k-NN query
    std::cout << "\nQuerying for vectors near (1.0, 1.0):" << std::endl;
    std::vector<hnsw::QueryResult> results = db.query({1.0f, 1.0f}, 2, nullptr, {hnsw::Include::ID, hnsw::Include::DISTANCE});
    for (const auto& result : results) {
        std::cout << "  - Found ID: " << result.id << " (Distance: " << result.distance << ")" << std::endl;
    }

    // 4. Delete a vector and update another
    std::cout << "\nDeleting vector 1 and updating vector 0..." << std::endl;
    db.delete_vector(1);
    uint32_t new_id_for_0 = db.update_vector(0, {1.5f, 1.5f}, {{"type", "A"}, {"updated", "true"}});
    std::cout << "  - Vector 0 now has new ID: " << new_id_for_0 << std::endl;

    // 5. Rebuild the index to apply changes permanently
    std::cout << "\nRebuilding index..." << std::endl;
    db.rebuild_index();

    // 6. Query again with a filter
    std::cout << "\nQuerying for type 'B' vectors near (5.0, 5.0):" << std::endl;
    auto filter_b = [](const hnsw::Metadata& meta) {
        return meta.count("type") && meta.at("type") == "B";
    };
    results = db.query({5.0f, 5.0f}, 2, filter_b, {hnsw::Include::ID});
    for (const auto& result : results) {
        // Note: IDs are re-assigned after rebuild_index()
        std::cout << "  - Found ID: " << result.id << std::endl;
    }

    // 7. Save to disk
    std::cout << "\nSaving database to " << db_path << std::endl;
    db.save();

    return 0;
}
```