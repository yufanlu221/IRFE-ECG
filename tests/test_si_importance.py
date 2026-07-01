import torch

from trainer.shared_baselines import SI_XI, finalize_si_importance


def test_finalize_si_uses_selected_epoch_path_and_anchor() -> None:
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(2.0)

    state = {
        "theta_before_task": {"weight": torch.zeros_like(model.weight)},
        "small_omega": {"weight": torch.full_like(model.weight, 100.0)},
    }
    selected_small_omega = {"weight": torch.full_like(model.weight, 4.0)}
    terms = []

    finalize_si_importance(
        model,
        state,
        terms,
        small_omega_override=selected_small_omega,
    )

    assert len(terms) == 1
    assert torch.equal(terms[0]["means"]["weight"], model.weight.detach().cpu())
    expected = torch.tensor([[4.0 / (4.0 + SI_XI)]])
    assert torch.allclose(terms[0]["fishers"]["weight"], expected)


def test_finalize_si_accumulates_previous_importance() -> None:
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)

    state = {
        "theta_before_task": {"weight": torch.zeros_like(model.weight)},
        "small_omega": {"weight": torch.ones_like(model.weight)},
    }
    terms = [
        {
            "means": {"weight": torch.zeros_like(model.weight)},
            "fishers": {"weight": torch.full_like(model.weight, 2.0)},
        }
    ]

    finalize_si_importance(model, state, terms)

    expected = torch.tensor([[2.0 + 1.0 / (1.0 + SI_XI)]])
    assert torch.allclose(terms[0]["fishers"]["weight"], expected)
