def model_dict_assign(**kwargs) -> dict:
    model_dict = {}
    for key, value in kwargs.items():
        model_dict[key] = value
    return model_dict
