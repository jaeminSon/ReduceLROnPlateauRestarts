# ReduceLROnPlateauRestarts

![image](https://user-images.githubusercontent.com/8290383/211699878-cf5a5aad-7a71-4aa1-ba87-e856f0437b48.png)

## Usage 
```
>>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
>>> scheduler = ReduceLROnPlateauRestarts(optimizer, min_lr=5e-5, max_lr=1e-1, patience=10)
>>> for epoch in range(10):
>>>     train(...)
>>>     val_loss = validate(...)
>>>     # Note that step should be called after validate()
>>>     scheduler.step(val_loss)
```

## Experiments
- data: CIFAR10
- augmentation: random crop (padding=4), horizontal flip
- optimizer: SGD
- lr: {max:1e-1, min=1e-4}
- codebase: [autolrs](https://github.com/YuchenJin/autolrs)

```validation accuracy by epoch```

![ReduceOnPlateauRestartsVal](https://user-images.githubusercontent.com/8290383/211694793-b84d2ec5-4ced-4aef-aced-2e1d26f935db.png) ![CosineAnnealingWithRestartsVal](https://user-images.githubusercontent.com/8290383/211694750-5481c364-6f2a-411d-8da1-ba3e613bf8f8.png)

```best validation accuracy by epoch```
   
<img src="https://user-images.githubusercontent.com/8290383/211695183-387da49b-bda8-45be-b045-9c083f70cc51.png"  width="267" height="191" /> <img src="https://user-images.githubusercontent.com/8290383/211695176-30f9fdd4-4b8d-408c-8c92-6adee22d6015.png"  width="267" height="191" />

```final best validation accuracy ```

<img src="https://user-images.githubusercontent.com/8290383/211695489-0c99ad6d-f8ed-4ba9-a70b-6c81b12c9e77.png"  width="267" height="191" /> <img src="https://user-images.githubusercontent.com/8290383/211695550-97509775-db46-47df-8d3c-bbd77cb1ca26.png"  width="267" height="191" />


## Comparison with other methods 

```ran the authors' code as is with CIFAR10```

| paper | method | use val metric| wallclock time (hours) | best val acc | specifics | code | 
| - | - | - | - | - | - | - |
| AUTOLRS: AUTOMATIC LEARNING-RATE SCHEDULE BY BAYESIAN OPTIMIZATION ON THE FLY  | Bayesian Optimization | YES |~1.3|93.7%|lr progression similar to cosine annealing|[link](https://github.com/YuchenJin/autolrs)|
| ONLINE LEARNING RATE ADAPTATION WITH HYPERGRADIENT DESCENT | Hypergradient | NO |~0.8|87.3%|negative lr, noisy peaks in lr|[link](https://github.com/gbaydin/hypergradient-descent)|
| Online hyperparameter optimization by real-time recurrent learning | Hypergradient + RNN training | YES |>11|<92.4%|heavy computes and memory demand(~7G for cifar10), lr converging to a certain value |[link](https://github.com/jiwoongim/OHO)|
|Learning an Adaptive Learning Rate Schedule| Reinforcement Learning (PPO) | YES | ~10 | <30% | heavy computes, sudden drop in val acc | [link](https://github.com/nicklashansen/adaptive-learning-rate-schedule)|
|CosineAnnealing without Restart|heuristic|NO|~0.6| 93.5%|period parameter|pytorch|
|CosineAnnealing with Restart|heuristic|NO|~0.6| 93.5%|period parameter|pytorch|
|ReduceLROnPlateauRestarts|heuristic|YES|~0.6| 93.4%|patience parameter|this repo|

Online hyperparameter optimization by real-time recurrent learning has 3 methods - global, full, layerwise
| method | wallclock time (hours) | best val acc |
| - | - | - |
| global | ~11 | ~88.5% |
| full | ~24 | 92.4% |
| layerwise | ~30 | 91.7%|

## Materials
optimizer comparison: "Descending through a Crowded Valley — Benchmarking Deep Learning Optimizers"
