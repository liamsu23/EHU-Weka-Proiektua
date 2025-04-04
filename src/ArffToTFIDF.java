import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.tokenizers.WordTokenizer;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;
import weka.filters.unsupervised.attribute.FixedDictionaryStringToWordVector;
import weka.filters.unsupervised.attribute.Reorder;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;

public class ArffToTFIDF {
    public static void main(String[] args) throws Exception {
        if (args.length != 7) {
            System.out.println("Uso: java ArffToTFIDF <train_RAW.arff> <dev_RAW.arff> <test_RAW.arff> <hiztegia.txt> <train_BOW.arff> <dev_BOW.arff> <test_BOW.arff>");
            return;
        }

        try {
            String inTrainRAWPath = args[0];
            String inDevRAWPath = args[1];
            String inTestRAWPath = args[2];
            String outDictionaryPath = args[3];
            String outTrainBOWPath = args[4];
            String outDevBOWPath = args[5];
            String outTestBOWPath = args[6];

            System.out.println("Procesando train_RAW.arff...");
            Instances trainBOWData = processTrainingData(inTrainRAWPath, outDictionaryPath, outTrainBOWPath);

            System.out.println("Procesando dev_RAW.arff...");
            processTestData(inDevRAWPath, outDevBOWPath, outDictionaryPath, trainBOWData);

            System.out.println("Procesando test_RAW.arff...");
            processTestData(inTestRAWPath, outTestBOWPath, outDictionaryPath, trainBOWData);

            System.out.println("Proceso finalizado con éxito.");
        } catch (Exception e) {
            e.printStackTrace();
            System.out.println("ERROR");
        }
    }

    /**
     * Procesa el conjunto de entrenamiento aplicando TF-IDF y guarda el diccionario.
     */
    private static Instances processTrainingData(String trainRAWPath, String dictionaryPath, String trainBOWPath) throws Exception {
        DataSource source = new DataSource(trainRAWPath);
        Instances trainRAWData = source.getDataSet();

        if (trainRAWData.classIndex() == -1) {
            trainRAWData.setClassIndex(trainRAWData.numAttributes() - 1);
        }

        // Aplicar TF-IDF y guardar el diccionario
        StringToWordVector filter = new StringToWordVector();
        filter.setTFTransform(true);  // Activa TF
        filter.setIDFTransform(true); // Activa IDF
        filter.setLowerCaseTokens(true);
        filter.setDictionaryFileToSaveTo(new File(dictionaryPath)); // Guarda el diccionario

        WordTokenizer tokenizer = new WordTokenizer();
        tokenizer.setDelimiters(" \r\n\t.,;:'\"()?![]@%&*/+-=<>{}^|#");
        filter.setTokenizer(tokenizer);

        filter.setInputFormat(trainRAWData);
        Instances trainBOWData = Filter.useFilter(trainRAWData, filter);

        trainBOWData = moveClassToLast(trainBOWData);

        // Guardar el archivo .arff con TF-IDF
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(trainBOWPath))) {
            writer.write(trainBOWData.toString());
        }

        return trainBOWData;
    }

    /**
     * Procesa el conjunto de validación/test aplicando TF-IDF con el diccionario guardado.
     */
    private static void processTestData(String dev_testRAWPath, String bowPath, String dictionaryPath, Instances trainBOWData) throws Exception {
        DataSource source = new DataSource(dev_testRAWPath);
        Instances dev_testRAWData = source.getDataSet();

        if (dev_testRAWData.classIndex() == -1) {
            dev_testRAWData.setClassIndex(dev_testRAWData.numAttributes() - 1);
        }

        // Aplicar TF-IDF usando el diccionario guardado
        FixedDictionaryStringToWordVector filter = new FixedDictionaryStringToWordVector();
        filter.setDictionaryFile(new File(dictionaryPath));
        filter.setLowerCaseTokens(true);
        filter.setTFTransform(true);  // Activar TF para coherencia con train
        filter.setIDFTransform(true); // Activar IDF para coherencia con train
        filter.setOutputWordCounts(false); // Obtener TF-IDF en lugar de conteos

        WordTokenizer tokenizer = new WordTokenizer();
        tokenizer.setDelimiters(" \r\n\t.,;:'\"()?![]@%&*/+-=<>{}^|#");
        filter.setTokenizer(tokenizer);

        filter.setInputFormat(dev_testRAWData);
        Instances bowData = Filter.useFilter(dev_testRAWData, filter);

        bowData = moveClassToLast(bowData);

        // Verificar que `dev` y `test` tienen los mismos atributos que `train`
        System.out.println("Verificando headers...");
        if (trainBOWData.numAttributes() != bowData.numAttributes()) {
            System.out.println("Advertencia: Diferente número de atributos en train y test.");
        } else {
            for (int i = 0; i < trainBOWData.numAttributes(); i++) {
                if (!trainBOWData.attribute(i).name().equals(bowData.attribute(i).name())) {
                    System.out.println("Advertencia: Diferencias en los nombres de atributos en posición " + i);
                }
            }
        }

        // Guardar el archivo .arff con TF-IDF
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(bowPath))) {
            writer.write(bowData.toString());
        }
    }

    /**
     * Mueve la clase al final del dataset.
     */
    private static Instances moveClassToLast(Instances data) throws Exception {
        if (data.classIndex() == -1) {
            throw new IllegalStateException("La clase no está definida.");
        }

        StringBuilder indices = new StringBuilder();
        for (int i = 0; i < data.numAttributes(); i++) {
            if (i != data.classIndex()) {
                indices.append(i + 1).append(",");
            }
        }
        indices.append(data.classIndex() + 1);

        Reorder reorderFilter = new Reorder();
        reorderFilter.setAttributeIndices(indices.toString());
        reorderFilter.setInputFormat(data);
        return Filter.useFilter(data, reorderFilter);
    }
}
