1. Number of unique labels for 'Lamp' class in sem-seg 
    - Level 1: [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17]
    - Level 2: [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27]
    - Level 3: [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40]
2. Shapenet Part to Shapenet v2 alignmen matrix: `trans_mtx = np.array([[0,0,1],[0,-1,0],[-1,0,0]])`. Usage:
``` python
aligned_shapenet_part = shapenet_part @ trans_mtx.T
```
3. Shapenet Part to Shapenet v2 steps:

```
shapenet_part = shapenet_part @ trans_mtx.T
shapenet_part = icp(shapenet_part, shapenet_v2) # calculate transform from part to v2 using icp
```

4. JSON Schema

- Shapnet ID
- Shapenet Cat ID
- Shapenet Cat Name
- Anno ID
- Shapenet 1 Path
- Shapenet 1 Meta
- Shapenet 2 Path
- Shapenet 2 Meta
- ShapenetSem Path
- ShapenetSem Meta
- Partnet Path
- v1 to v2
- v1 to v2 norm
- v1 to v2 align
- v1 to v2 loss
- sem to v2
- sem to v2 norm
- sem to v2 align
- sem to v2 loss
- partnet to v2
- partnet to v2 norm
- partnet to v2 align
- partnet to v2 loss