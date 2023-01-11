import pytest

from scheduler import ReduceLROnPlateauRestarts


class Test:
    @pytest.mark.parametrize("min_lr,max_lr,patience", [(5e-7,1e-4,5)])
    def test_steady_val_measure_lr_list(self, min_lr, max_lr, patience):
        
        import torch
        from torchvision.models import resnet18

        optimizer = torch.optim.AdamW(resnet18().parameters(), max_lr)
        n_param_groups = len(optimizer.param_groups)
        lr_scheduler = ReduceLROnPlateauRestarts(optimizer, [min_lr]*n_param_groups, [max_lr]*n_param_groups, patience=patience)
        
        list_lr = []
        for _ in range(100):
            lr_scheduler.step(0)
            list_lr.append(lr_scheduler.optimizer.param_groups[0]['lr'])

        expected = [max_lr*(0.1)**((i//6)%3) for i in range(100)] # [1e-4]*6 + [1e-5]*6 + [1e-6]*6 + [1e-4]*6
        assert all([abs(list_lr[i]-expected[i]) < 1e-8 for i in range(100)])

    @pytest.mark.parametrize("min_lr,max_lr,patience", [(5e-7,1e-4,5)])
    def test_improving_val_measure_lr_list(self, min_lr, max_lr, patience):
        
        import torch
        from torchvision.models import resnet18

        optimizer = torch.optim.AdamW(resnet18().parameters(), max_lr)
        n_param_groups = len(optimizer.param_groups)
        lr_scheduler = ReduceLROnPlateauRestarts(torch.optim.AdamW(resnet18().parameters(), max_lr), [min_lr]*n_param_groups, [max_lr]*n_param_groups, patience=patience)

        list_lr = []
        for i in range(100):
            lr_scheduler.step(-i)
            list_lr.append(lr_scheduler.optimizer.param_groups[0]['lr'])

        expected = [max_lr] * 100 # [1e-4]*100
        assert all([abs(list_lr[i]-expected[i]) < 1e-8 for i in range(100)])

    @pytest.mark.parametrize("min_lr,max_lr,patience", [(5e-7,1e-4,5)])
    def test_steady_val_measure(self, min_lr, max_lr, patience):
        
        import torch
        from torchvision.models import resnet18

        lr_scheduler = ReduceLROnPlateauRestarts(torch.optim.AdamW(resnet18().parameters(), max_lr), min_lr, max_lr, patience=patience)

        list_lr = []
        for _ in range(100):
            lr_scheduler.step(0)
            list_lr.append(lr_scheduler.optimizer.param_groups[0]['lr'])

        expected = [max_lr*(0.1)**((i//6)%3) for i in range(100)] # [1e-4]*6 + [1e-5]*6 + [1e-6]*6 + [1e-4]*6
        assert all([abs(list_lr[i]-expected[i]) < 1e-8 for i in range(100)])

    @pytest.mark.parametrize("min_lr,max_lr,patience", [(5e-7,1e-4,5)])
    def test_improving_val_measure(self, min_lr, max_lr, patience):
        
        import torch
        from torchvision.models import resnet18

        lr_scheduler = ReduceLROnPlateauRestarts(torch.optim.AdamW(resnet18().parameters(), max_lr), min_lr, max_lr, patience=patience)

        list_lr = []
        for i in range(100):
            lr_scheduler.step(-i)
            list_lr.append(lr_scheduler.optimizer.param_groups[0]['lr'])

        expected = [max_lr] * 100 # [1e-4]*100
        assert all([abs(list_lr[i]-expected[i]) < 1e-8 for i in range(100)])