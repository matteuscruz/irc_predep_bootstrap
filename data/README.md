# Dados de Entrada

Esta pasta deve conter os dados necessários para executar as análises PREDEP.

## Estrutura Esperada

```
data/
├── input/
│   ├── merged_indices.csv                           # Índices MoV mensais
│   └── seasonal_precipitation_anomalies_1982_2020.nc  # Dados de precipitação sazonal
├── processed/                                       # Dados processados (gerados)
└── raw/                                            # Dados brutos (opcional)
```

## Arquivos Necessários

### 1. merged_indices.csv
- **Descrição**: Índices de Modos de Variabilidade (MoV) mensais
- **Formato**: CSV com colunas para cada índice MoV
- **Período**: 1982-2020 (ou conforme disponível)
- **Colunas esperadas**: ATL3, NAO, PDO, etc.

### 2. seasonal_precipitation_anomalies_1982_2020.nc
- **Descrição**: Anomalias de precipitação sazonal
- **Formato**: NetCDF com dimensões [time, latitude, longitude]
- **Período**: 1982-2020
- **Variável**: 'pr' (precipitação)

## Como Obter os Dados

1. **Contate a equipe do projeto** para acesso aos dados originais
2. **Coloque os arquivos** na pasta `data/input/`
3. **Verifique a estrutura** executando:
   ```bash
   .venv/bin/python -c "
   import pandas as pd
   import xarray as xr
   mov = pd.read_csv('data/input/merged_indices.csv')
   precip = xr.open_dataset('data/input/seasonal_precipitation_anomalies_1982_2020.nc')
   print('MoVs:', list(mov.columns))
   print('Precipitação:', precip.dims)
   "
   ```

## Dados de Exemplo

Para testes, você pode gerar dados sintéticos:

```python
import pandas as pd
import xarray as xr
import numpy as np

# Gerar índices MoV sintéticos
dates = pd.date_range('1982-01-01', '2020-12-31', freq='M')
mov_data = pd.DataFrame({
    'ATL3': np.random.normal(0, 1, len(dates)),
    'NAO': np.random.normal(0, 1, len(dates)),
    'PDO': np.random.normal(0, 1, len(dates))
})
mov_data.to_csv('data/input/merged_indices.csv', index=False)

# Gerar dados de precipitação sintéticos
time_seasonal = pd.date_range('1982-03-31', '2020-12-31', freq='3M')
precip_data = xr.Dataset({
    'pr': (['time', 'latitude', 'longitude'], 
           np.random.normal(0, 1, (len(time_seasonal), 10, 10)))
}, coords={
    'time': time_seasonal,
    'latitude': np.linspace(-10, 10, 10),
    'longitude': np.linspace(-50, -40, 10)
})
precip_data.to_netcdf('data/input/seasonal_precipitation_anomalies_1982_2020.nc')
```

## Nota Importante

⚠️ **Os dados não são versionados no Git** por questões de tamanho e privacidade. Certifique-se de ter os dados localmente antes de executar os experimentos.