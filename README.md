# Ataque via GAN em Aprendizado Federado utilizando Flower Framework

Reprodução do artigo **"Deep Models Under the GAN: Information Leakage from
Collaborative Deep Learning"** ([arXiv:1702.07464](https://arxiv.org/abs/1702.07464))
feito com **Flower 1.25.x** + **PyTorch** utilizando o dataset **MNIST**.

## Layout do projeto

```
gan_attack_flower/
├── gan_attack_flower/
│   ├── __init__.py
│   ├── task.py         # modelos, dados treinamento
│   ├── attack.py       # ataque com treinamento via GAN
│   ├── client_app.py   # Cliente Honesto + Cliente Malicioso
│   └── server_app.py   # FedAvg ServerApp
├── pyproject.toml
└── README.md
```
