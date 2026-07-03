# MODELS COMPARISON

## Comparison PaddleOCR v5 vs v6


| Model          | Detection size (Mo)| Recognition size (Mo)| Total size (Mo)| average time (s/image) (DEBUG_IMAGE = True) | Notes |
|----------------|--------------------|----------------------|----------------|---------------------------------------------|-------|
| v5 server      |                    |                      |                |                                          |    |
| v5 server ONNX |                    |                      |                |                                          |   |
| v5 mobile      |4.9                 |8.2                   |13.1            | 1271.60/386 = 3.3                           | all tests passed  |
| v5 mobile ONNX |4.8                 |8.0                   |12.8            | 287.10/386 = 0.74                          | 1 test failed |
| v6 medium      |62.3                |76.8                  |139.1           | 3190.35/386 = 8.26                          | all tests passed |
| v6 medium ONNX |62.4                |77.0                  |139.4           | 1533.54/386 = 3.97                          | 2 tests failed |
| v6 small       |10.1                |21.4                  |31.5            | 939.33/386 = 2.43                           | 1 test failed |
| v6 small ONNX  |10.1                |21.5                  |31.6            | 330.83/386 = 0.86                           | 4 tests failed |
| v6 tiny        |2.0                 |4.6                   |6.6             | 435.28/386 = 1.13                           | 20 tests failed |
| v6 tiny ONNX   |2.0                 |4.6                   |6.6             | 171.82/386 = 0.45                           | 22 tests failed |
