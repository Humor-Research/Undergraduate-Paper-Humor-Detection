# Undergraduate Paper Humor Detection

Дополнительные файлы для ВКР, не вошедшие в основной репозиторий:

additional_datasets.ipynb – формирование дополнительных датасетов, упомянутых в оригинальной статье.

conversational_datasets.ipynb – формирование новых дополнительных датасетов, впервые представленных в ВКР.

attack_results.ipynb – примеры, численные результаты и облака слов для новых состязательных атак.


run_attack_input_reduction.py, run_attack_input_reduction.sbatch – файлы, связанные с атакой "Ключевые слова". 

run_attack_symbols_change.py, run_attack_symbols_change.sbatch – файлы, связанные с атакой "Изменение символов". 

Запуск состязательных атак на SLURM сервере:

```
sbatch run_attack_{$ATTACK_NAME}.sbatch
```

Для запуска Python-файла:

```
python run_attack_{$ATTACK_NAME}.py $TEST_DATASET
```



