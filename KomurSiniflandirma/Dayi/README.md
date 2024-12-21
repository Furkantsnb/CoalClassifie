# Test Açıklamaları

Bu testler, bir Python betiğinin (`main.py` olarak adlandırılan) işlevselliğini test etmek için yazılmıştır. İşlevler `dataset_loading` ve `build_efficentB0` olmak üzere iki ana bölüme ayrılmıştır.

## Örnek Veri Seti

`sample_data` adında bir `pytest.fixture` tanımlanmıştır. Bu, diğer test fonksiyonlarında kullanılmak üzere bir örnek veri seti sağlar. Bu veri seti, `dataset_loading` işlevi kullanılarak elde edilir.

## dataset_loading Testi

`test_dataset_loading` işlevi, `dataset_loading` işlevinin doğru şekilde çalışıp çalışmadığını kontrol eder. Örneğin, bu test veri kümelerinin TensorFlow veri kümeleri olduğunu doğrular.

## build_efficentB0 Testi

`test_build_efficentB0` işlevi, `build_efficentB0` işlevinin beklenen şekilde bir model döndürüp döndürmediğini kontrol eder. Bu test, dönen nesnenin bir TensorFlow Keras Modeli olduğunu doğrular.

## Model Eğitim Testi

`test_model_training` işlevi, modelin eğitim işlevini test eder. Bu, modelin belirli bir eğitim setiyle eğitilebilir ve doğru bir şekilde geçmiş (history) nesnesi döndürdüğünü kontrol eder.

## Model Değerlendirme Testi

`test_model_evaluation` işlevi, modelin değerlendirme işlevini test eder. Bu, modelin belirli bir test seti üzerinde değerlendirilebildiğini doğrular.

Bu testler, betiğin doğru çalıştığından emin olmak için yazılmıştır. Eğer bir test başarısız olursa, işlevlerde veya veri setlerinde bir hata olabilir ve bu hataların düzeltilmesi gerekebilir.