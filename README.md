# PerAct-MISO
Optimizing PerAct: [https://peract.github.io/] using MISO mapping: [https://existentialrobotics.org/miso_rss25/].

Run this for installing the grid optimizer (grid_opt):

```
pip install -e mod/MISO
```

Then install RLBench:
```
pip install -e mod/RLBench
```

To build, run:
```
docker build -t peract-miso-base -f Dockerfile.base .
```
```
docker build -t peract-miso-dev -f Dockerfile.dev .
```

Then to run without GPUs:
```
docker run -it --rm -v $(pwd):/workspace peract-dev /bin/bash
```

Or this to run with GPUs:
```
docker run -it --rm --gpus all -v $(pwd):/workspace peract-dev /bin/bash
```


TODO:
MISO assumes a system that runs linux+CUDA. Is it possible to develop on macos and verify correctness?


NOTES:
Making fake X server to bypass pyrender import issues:
```
apt-get update && apt-get install -y xvfb
```
