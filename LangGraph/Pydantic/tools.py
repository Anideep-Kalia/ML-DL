from schemas import WeatherRequest

def get_weather(req: WeatherRequest):
    # fake API
    return f"The weather in {req.city} is 32° {req.units}"