from omegaconf import DictConfig

import hydra

@hydra.main(config_path='conf', config_name='config')
def main(cfg: DictConfig) -> None:
    # Configuration check

    assert cfg.algorithm in ['GAIL', 'RED']
    print(cfg)


if __name__ == '__main__':
    main()
