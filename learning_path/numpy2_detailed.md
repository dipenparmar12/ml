## 1. Understanding NumPy Arrays

**Summary:** Meet the incredible NumPy array! Learn how to create and change array shapes to suit your needs. Finally, discover NumPy's many data types and how they contribute to speedy array operations.

**Examples & Info:**

* **Introducing arrays:**
    * **Info:** Explain what an array is in the context of NumPy (a grid of values, all of the same type, indexed by a tuple of non-negative integers). Contrast it with Python lists (heterogeneous, less efficient for numerical operations).
    * **Example:** Basic conceptualization of data organized in rows and columns, like a spreadsheet.
* **Your first NumPy array:**
    * **Info:** Introduce `np.array()`.
    * **Example:** `my_list = [1, 2, 3, 4]`, then `my_array = np.array(my_list)`. Show `print(my_array)`, `print(type(my_array))`.
* **Creating arrays from scratch:**
    * **Info:** Focus on `np.zeros()`, `np.ones()`, `np.full()`, `np.empty()`. Explain when each is useful (e.g., initializing an array of zeros for a computation, creating a placeholder array).
    * **Example:**
        * `zeros_array = np.zeros((3, 4))`
        * `ones_array = np.ones(5)`
        * `full_array = np.full((2, 2), 7)`
        * `empty_array = np.empty((2, 3))` (emphasize uninitialized values)
* **A range array:**
    * **Info:** Introduce `np.arange()` for creating arrays with a sequence of numbers, similar to Python's `range()`. Also, `np.linspace()` for evenly spaced numbers over a specified interval.
    * **Example:**
        * `range_array = np.arange(0, 10, 2)` (output: `[0, 2, 4, 6, 8]`)
        * `linspace_array = np.linspace(0, 1, 5)` (output: `[0.0, 0.25, 0.5, 0.75, 1.0]`)
* **Array dimensionality:**
    * **Info:** Explain "dimensions" (axes) and "rank" of an array. Show how `ndim` and `shape` attributes work.
    * **Example:**
        * `scalar = np.array(5)` (0-D array, `ndim=0`, `shape=()`)
        * `vector = np.array([1, 2, 3])` (1-D array, `ndim=1`, `shape=(3,)`)
        * `matrix = np.array([[1, 2], [3, 4]])` (2-D array, `ndim=2`, `shape=(2, 2)`)
* **3D array creation:**
    * **Info:** Extend the concept of `np.array()` to 3 dimensions, thinking of it as a "stack" of 2D arrays.
    * **Example:** `three_d_array = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])`. Show its `shape` and `ndim`.
* **The fourth dimension:**
    * **Info:** Briefly explain the concept of higher dimensions, even if not commonly used in everyday examples, to solidify the abstract understanding. Mention image processing (batches of images, height, width, color channels) or video data as potential use cases.
    * **Example:** Conceptually `(batch_size, frames, height, width, channels)` for video data. No complex code needed, just a verbal explanation.
* **Flattening and reshaping:**
    * **Info:** Explain `flatten()`, `ravel()`, and `reshape()`. Differentiate `flatten()` (returns a copy) and `ravel()` (returns a view). Emphasize `-1` in `reshape()` for automatic dimension calculation.
    * **Example:**
        * `arr_2d = np.array([[1, 2], [3, 4]])`
        * `flat_arr = arr_2d.flatten()`
        * `reshaped_arr = arr_2d.reshape(4, 1)` or `arr_2d.reshape(-1, 1)`
        * `reshaped_to_1d = arr_2d.reshape(-1)`
* **NumPy data types:**
    * **Info:** Discuss the importance of data types (e.g., `int32`, `float64`, `bool`). Explain how they affect memory usage and computational speed. Show `dtype` attribute.
    * **Example:**
        * `int_array = np.array([1, 2, 3])` (default `int64` or `int32` depending on system)
        * `float_array = np.array([1.0, 2.5])` (default `float64`)
        * `bool_array = np.array([True, False])`
        * Show `print(int_array.dtype)` etc.
* **The `dtype` argument:**
    * **Info:** Show how to explicitly set the data type during array creation using the `dtype` argument.
    * **Example:** `specified_dtype_array = np.array([1, 2, 3], dtype=np.float32)`
* **Anticipating data types:**
    * **Info:** Discuss common scenarios where NumPy infers data types and potential issues (e.g., mixing integers and floats results in floats, strings make everything strings).
    * **Example:**
        * `mixed_type_array = np.array([1, 2.5, 3])` (will be `float64`)
        * `string_in_array = np.array([1, 'hello', 3])` (will be `U5` or similar string type)
* **A smaller sudoku game:**
    * **Info:** A fun, practical application of array creation and initial understanding.
    * **Example:** Create a partially filled 3x3 or 4x4 Sudoku grid using `np.zeros()` for empty cells and `np.array()` for known numbers. `sudoku_grid = np.array([[0, 1, 0], [2, 0, 3], [0, 4, 0]])`

---

## 2. Selecting and Updating Data

**Summary:** Sharpen your NumPy data wrangling skills by slicing, filtering, and sorting New York City’s tree census data. Create new arrays by pulling data based on conditional statements, and add and remove data along any dimension to suit your purpose. Along the way, you’ll learn the shape and dimension compatibility principles to prepare for super-fast array math.

**Examples & Info:**

* **Indexing and slicing arrays:**
    * **Info:** Explain 0-based indexing. Demonstrate single element access, and basic slicing `[start:end:step]`. Highlight that slicing returns a view, not a copy (by default).
    * **Example:**
        * `data = np.array([10, 20, 30, 40, 50])`
        * `data[0]` (10)
        * `data[1:4]` (20, 30, 40)
        * `data[::-1]` (reversed array)
* **Slicing and indexing trees:**
    * **Info:** Apply 2D slicing to a hypothetical dataset.
    * **Example:** Imagine a `tree_data` array `(rows: trees, columns: ['height', 'diameter', 'age'])`.
        * `tree_data = np.array([[10, 5, 20], [12, 6, 25], [8, 4, 15]])`
        * `tree_data[0, 1]` (diameter of the first tree)
        * `tree_data[:, 0]` (all tree heights)
        * `tree_data[1:3, :]` (second and third trees, all data)
* **Stepping into 2D:**
    * **Info:** Elaborate on using `[row_slice, col_slice]` with steps.
    * **Example:** `matrix = np.arange(1, 17).reshape(4, 4)`
        * `matrix[::2, ::2]` (every other row, every other column)
* **Sorting trees:**
    * **Info:** Introduce `np.sort()` (returns a sorted copy) and `.sort()` method (sorts in-place). Explain sorting along axes.
    * **Example:**
        * `unsorted_heights = np.array([12, 8, 10, 15])`
        * `sorted_heights = np.sort(unsorted_heights)`
        * `tree_data_sorted_by_height = tree_data[np.argsort(tree_data[:, 0])]` (sorting the whole array based on one column)
* **Filtering arrays:**
    * **Info:** Explain boolean indexing (masking). Create a boolean array based on a condition and use it to select elements.
    * **Example:** `ages = np.array([20, 25, 15, 30, 18])`
        * `ages > 20` (returns `[False, True, False, True, False]`)
        * `ages[ages > 20]` (returns `[25, 30]`)
* **Filtering with masks:**
    * **Info:** Deep dive into how masks work with multi-dimensional arrays.
    * **Example:** `tree_data[tree_data[:, 2] > 20]` (select trees older than 20 years)
* **Fancy indexing vs. `np.where()`:**
    * **Info:**
        * **Fancy indexing:** Using an array of integer indices to select elements. Returns a copy.
        * **`np.where()`:** Returns indices of elements satisfying a condition. Useful for conditional assignments or when you need the *locations*.
    * **Example:**
        * **Fancy indexing:** `data = np.array(['apple', 'banana', 'cherry', 'date'])`; `indices = np.array([0, 2, 2])`; `data[indices]` (['apple', 'cherry', 'cherry'])
        * **`np.where()`:** `prices = np.array([10, 5, 12, 8, 15])`; `cheap_indices = np.where(prices < 10)` (returns `(array([1, 3]),)`)
        * `np.where(prices < 10, 'cheap', 'expensive')` (returns `['expensive', 'cheap', 'expensive', 'cheap', 'expensive']`)
* **Creating arrays from conditions:**
    * **Info:** Combine filtering with assignments to create new arrays or modify existing ones based on conditions.
    * **Example:** `sales = np.array([100, 150, 75, 200, 120])`
        * `high_sales = sales[sales > 100]`
        * `sales[sales < 100] = 90` (update values)
* **Adding and removing data:**
    * **Info:** Introduce `np.append()`, `np.insert()`, `np.delete()`. Emphasize that these operations typically return a *new* array, as NumPy arrays are fixed-size.
    * **Example:** `arr = np.array([1, 2, 3])`
        * `np.append(arr, 4)`
        * `np.insert(arr, 1, 99)`
* **Compatible or not?**
    * **Info:** Introduce the concept of shape compatibility for operations like `np.append()` and `np.insert()`. Explain that dimensions must match or be compatible for the operation to succeed.
    * **Example:** Attempting to `np.append` a 1D array to a 2D array without specifying an axis.
* **Adding rows:**
    * **Info:** Use `np.append()` or `np.vstack()` with `axis=0` to add rows to a 2D array.
    * **Example:**
        * `matrix = np.array([[1, 2], [3, 4]])`
        * `new_row = np.array([[5, 6]])`
        * `matrix_with_new_row = np.append(matrix, new_row, axis=0)` or `np.vstack((matrix, new_row))`
* **Adding columns:**
    * **Info:** Use `np.append()` or `np.hstack()` with `axis=1` to add columns.
    * **Example:**
        * `matrix = np.array([[1, 2], [3, 4]])`
        * `new_col = np.array([[5], [6]])`
        * `matrix_with_new_col = np.append(matrix, new_col, axis=1)` or `np.hstack((matrix, new_col))`
* **Deleting with `np.delete()`:**
    * **Info:** Explain how to delete elements, rows, or columns using `np.delete()`, specifying the index and axis.
    * **Example:**
        * `data = np.array([10, 20, 30, 40])`
        * `np.delete(data, 1)` (deletes 20)
        * `matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])`
        * `np.delete(matrix, 0, axis=0)` (deletes the first row)
        * `np.delete(matrix, 2, axis=1)` (deletes the third column)

---

## 3. Array Mathematics!

**Summary:** Leverage NumPy’s speedy vectorized operations to gather summary insights on sales data for American liquor stores, restaurants, and department stores. Vectorize Python functions for use in your NumPy code. Finally, use broadcasting logic to perform mathematical operations between arrays of different sizes.

**Examples & Info:**

* **Summarizing data:**
    * **Info:** Introduce common aggregation functions: `sum()`, `mean()`, `median()`, `min()`, `max()`, `std()` (standard deviation), `var()` (variance). Explain the `axis` argument for aggregations across specific dimensions.
    * **Example:**
        * `sales_data = np.array([[100, 120, 150], [80, 90, 110], [200, 180, 210]])` (rows: stores, columns: months)
        * `np.sum(sales_data)` (total sales)
        * `np.mean(sales_data, axis=0)` (average sales per month)
        * `np.max(sales_data, axis=1)` (max sales per store)
* **Sales totals:**
    * **Info:** Practical application of `sum()` or `np.sum()`.
    * **Example:** Calculate the total sales for a quarter or a year given a sales array.
        * `quarterly_sales = np.array([1000, 1200, 1150])`
        * `total = np.sum(quarterly_sales)`
* **Plotting averages:**
    * **Info:** Combine `mean()` with a conceptual link to data visualization (e.g., using Matplotlib, even if not directly coding the plot in NumPy).
    * **Example:** Given daily temperature readings, calculate weekly averages and discuss how you'd plot them over time to see trends.
* **Cumulative sales:**
    * **Info:** Introduce `np.cumsum()` for calculating cumulative sums along an axis.
    * **Example:** `daily_sales = np.array([50, 75, 60, 90, 110])`
        * `cumulative_daily_sales = np.cumsum(daily_sales)`
* **Vectorized operations:**
    * **Info:** Explain the core concept of vectorized operations in NumPy – applying operations element-wise without explicit loops, leading to significant speed improvements. Contrast with traditional Python loops.
    * **Example:**
        * `arr1 = np.array([1, 2, 3])`, `arr2 = np.array([4, 5, 6])`
        * `arr1 + arr2` (element-wise addition)
        * `arr1 * 2` (scalar multiplication)
        * `np.sqrt(arr1)`
* **Tax calculations:**
    * **Info:** Practical application of vectorized operations for financial calculations.
    * **Example:** `prices = np.array([10.50, 20.00, 5.75])`
        * `tax_rate = 0.08`
        * `prices_with_tax = prices * (1 + tax_rate)`
* **Projecting sales:**
    * **Info:** More complex vectorized calculations involving multiple arrays or conditions.
    * **Example:** If sales increase by a certain percentage month-over-month.
        * `current_sales = np.array([1000, 1500, 1200])`
        * `growth_factors = np.array([1.05, 1.07, 1.03])`
        * `projected_sales = current_sales * growth_factors`
* **Vectorizing `.upper()`:**
    * **Info:** Explain `np.vectorize()` for applying a non-NumPy (scalar) function element-wise to a NumPy array. Emphasize that it's generally slower than native NumPy ufuncs but useful for string operations or custom functions.
    * **Example:**
        * `names = np.array(['alice', 'bob', 'charlie'])`
        * `uppercase_func = np.vectorize(str.upper)`
        * `uppercase_names = uppercase_func(names)` (output: `['ALICE', 'BOB', 'CHARLIE']`)
* **Broadcasting:**
    * **Info:** Explain the rules of broadcasting: when arrays have different shapes, NumPy attempts to "broadcast" the smaller array across the larger array so that they have compatible shapes. Rules: 1. If dimensions differ, prepend 1s to the smaller shape. 2. Compare dimension sizes. They must be equal, or one must be 1. 3. If a dimension is 1 in one array and greater than 1 in the other, the array with 1 is stretched.
    * **Example:** Conceptualize adding a 1D array to each row of a 2D array.
* **Broadcastable or not?**
    * **Info:** Provide examples of shapes that are compatible and incompatible for broadcasting, walking through the rules.
    * **Example:**
        * `(3, 4) + (4,)` (Yes, `(4,)` becomes `(1, 4)`, then stretched)
        * `(3, 4) + (3,)` (No, incompatible at last dimension, `(3,)` becomes `(1, 3)`)
        * `(2, 3, 4) + (4,)` (Yes)
        * `(2, 3, 4) + (3, 1)` (No)
* **Broadcasting across columns:**
    * **Info:** Apply broadcasting rules to add a 1D array (representing a column) to each column of a 2D array. This often involves `reshape(-1, 1)` to make the 1D array a column vector.
    * **Example:**
        * `matrix = np.array([[1, 2, 3], [4, 5, 6]])`
        * `col_offset = np.array([10, 20]).reshape(-1, 1)` (becomes `[[10], [20]]`)
        * `result = matrix + col_offset`
* **Broadcasting across rows:**
    * **Info:** Apply broadcasting rules to add a 1D array (representing a row) to each row of a 2D array. This is often more straightforward.
    * **Example:**
        * `matrix = np.array([[1, 2, 3], [4, 5, 6]])`
        * `row_offset = np.array([10, 20, 30])`
        * `result = matrix + row_offset`

---

## 4. Array Transformations

**Summary:** NumPy meets the art world in this final chapter as we use image data from a Monet masterpiece to explore how you can use NumPy to augment image data. You’ll use flipping and transposing functionality to quickly transform our masterpiece. Next, you’ll pull the Monet array apart, make changes, and reconstruct it using array stacking to see the results.

**Examples & Info:**

* **Saving and loading arrays:**
    * **Info:** Introduce `np.save()` and `np.load()` for efficient binary storage of NumPy arrays (`.npy` files). Mention `np.savetxt()` and `np.loadtxt()` for text files (CSV, TSV), noting they are slower and less space-efficient for large arrays but human-readable.
    * **Example:**
        * `data_to_save = np.arange(100)`
        * `np.save('my_array.npy', data_to_save)`
* **Loading `.npy` files:**
    * **Info:** Show how to load previously saved `.npy` files.
    * **Example:** `loaded_data = np.load('my_array.npy')`
* **Getting help:**
    * **Info:** Demonstrate `np.info()` or `help(np.function_name)` for quick documentation access within a Python environment.
    * **Example:** `np.info(np.save)`
* **Update and save:**
    * **Info:** Combine loading, modifying an array, and then saving the updated version.
    * **Example:**
        * Load `sales_data.npy`.
        * Add a new month of sales or adjust existing values.
        * `np.save('updated_sales_data.npy', sales_data)`
* **Array acrobatics:**
    * **Info:** Broad overview of operations that change the arrangement of array elements without changing the data itself (e.g., flipping, transposing, rotating).
    * **Example:** Conceptually showing how an image might be rotated or mirrored.
* **Augmenting Monet:**
    * **Info:** Use image data (represented as a NumPy array of pixel values, typically `(height, width, channels)`) as a compelling real-world example for transformations.
    * **Example:** Load a simple grayscale image (or represent one as a 2D array, e.g., `monet_image = np.zeros((100, 100), dtype=np.uint8)` and fill some pixels).
* **Transposing your masterpiece:**
    * **Info:** Explain `np.transpose()` or the `.T` attribute for swapping axes.
    * **Example:**
        * `image_data = np.array([[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [100, 100, 100]]])` (2x2 image with RGB channels)
        * `transposed_image = image_data.T` or `np.transpose(image_data, (1, 0, 2))` (swapping height and width)
        * Also `np.flipud()` (flip up/down) and `np.fliplr()` (flip left/right) for image-specific transformations.
* **Stacking and splitting:**
    * **Info:** Introduce `np.concatenate()`, `np.vstack()`, `np.hstack()`, `np.dstack()` for combining arrays, and `np.split()`, `np.vsplit()`, `np.hsplit()`, `np.dsplit()` for dividing arrays. Emphasize `axis` for concatenation/splitting.
    * **Example:**
        * `arr1 = np.array([[1, 2], [3, 4]])`
        * `arr2 = np.array([[5, 6], [7, 8]])`
        * `np.vstack((arr1, arr2))` (stacking rows)
        * `np.hstack((arr1, arr2))` (stacking columns)
* **2D split and stack:**
    * **Info:** Practical application of splitting and stacking 2D arrays.
    * **Example:** Divide a matrix into two halves and then stack them differently.
        * `matrix = np.arange(1, 26).reshape(5, 5)`
        * `left_half, right_half = np.hsplit(matrix, [2])` (split after 2 columns)
        * `recombined = np.vstack((left_half, right_half))` (recombining differently)
* **Splitting RGB data:**
    * **Info:** Show how to split a 3D image array (height, width, channels) into separate arrays for each color channel (Red, Green, Blue).
    * **Example:**
        * `monet_rgb = np.random.randint(0, 256, size=(100, 150, 3), dtype=np.uint8)` (simulated image)
        * `red_channel, green_channel, blue_channel = np.split(monet_rgb, 3, axis=2)`
* **Stacking RGB data:**
    * **Info:** Demonstrate reconstructing an image from individual color channels, useful after applying transformations or filters to individual channels.
    * **Example:**
        * Take the split channels from the previous example.
        * Apply some transformation (e.g., `red_channel = red_channel * 0.5`).
        * `recombined_monet = np.dstack((red_channel, green_channel, blue_channel))` or `np.concatenate((red_channel, green_channel, blue_channel), axis=2)`