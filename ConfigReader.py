import json


class ConfigReader:
    def __init__(self, config_file):
        with open(config_file, 'r') as file:
            self.config = json.load(file)

    def get_config(self):
        return self.config


if __name__ == "__main__":
    config = ConfigReader("ConfigFile.json")
    config_file = config.get_config()
    print(config_file)

