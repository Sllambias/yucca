|      GPU     |  CPUS/WORKERS  | SECONDS/EPOCH |    TASK   |    SETUP    |     Patch Size    |
|              |                |               |           |             |                   | # Leave this line empty as a copy-pasta template
|   A100 80GB  |   24/24        |      97       |    06     |  Default 3D |  [128, 128, 128]  |
|   A100 80GB  |   20/24        |      97       |    06     |  Default 3D |  [128, 128, 128]  |
|   A100 80GB  |   12/24        |      110      |    06     |  Default 3D |  [128, 128, 128]  |
|   A100 80GB  |   6/24         |      160      |    06     |  Default 3D |  [128, 128, 128]  |
|   A100 80GB  |   48/48        |      89       |    06     |  Default 3D |  [128, 128, 128]  | # First epoch ~400 seconds to start all these workers
|   A100 80GB  |   96/96        |      87       |    06     |  Default 3D |  [128, 128, 128]  | # First epoch ~400 seconds to start all these workers
|   A100 80GB  |   23/24        |      93       |    01     |  Default 3D |  [128, 128, 128]  |
|   A100 80GB  |   23/24        |      94       |    01     |  Default 3D |  [128, 128, 128]  | # Tested torch.set_float32_matmul_precision('medium') 
