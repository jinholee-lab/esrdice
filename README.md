# One and only 주의사항

**main 함수에서 scalar input output version이 Vector로 대체**


```python
elif config.mode == "AET":
    # from AET import FiniteAET
    # agent = FiniteAET(config, device=config.device)
    from AET import FiniteAET_Vector
    print("Initializing Vector form")
    agent = FiniteAET_Vector(config, device=config.device)
