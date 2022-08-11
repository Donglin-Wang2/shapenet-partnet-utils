1. Provide various tests for the library
2. Use more robust data storage format, such a H5 or SQL Lite so that we can query the data more easily
3. Figuring out a way to pickle and un-pickle the record in back-compatible way:
    - If there is an addition to the data classes, the fields are automatically populated