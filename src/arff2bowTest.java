import weka.classifiers.Classifier;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.FixedDictionaryStringToWordVector;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;

public class arff2bowTest {
    public static void main(String[] args) throws Exception {
        if (args.length < 4) {
            System.out.println("Erabilera: java arff2bowTest <test_raw.arff> <test_BoW.arff> <dictionary.txt> <train_BoW.arff>");
            return;
        }

        try {
            String testRawPath = args[0];      // Ruta al archivo test_raw.arff
            String testBoWPath = args[1];      // Ruta para guardar test_BoW.arff
            String dictionaryPath = args[2];  // Ruta al diccionario guardado
            String trainBoWPath = args[3];    // Ruta al archivo train_BoW.arff (para verificar headers)

            // 1. Cargar el conjunto de prueba
            System.out.println("Test datuak kargatzen: " + testRawPath);
            ConverterUtils.DataSource testSource = new ConverterUtils.DataSource(testRawPath);
            Instances testData = testSource.getDataSet();
            if (testData.classIndex() == -1) testData.setClassIndex(0); // Ajustar índice de la clase

            // 2. Aplicar FixedDictionaryStringToWordVector
            System.out.println("FixedDictionaryStringToWordVector aplikatzen...");
            FixedDictionaryStringToWordVector filter = new FixedDictionaryStringToWordVector();
            filter.setDictionaryFile(new File(dictionaryPath));
            filter.setOutputWordCounts(false);
            filter.setInputFormat(testData);

            Instances testBoW = Filter.useFilter(testData, filter);

            // 3. Guardar test_BoW.arff
            System.out.println("Test datu sparse gordetzen: " + testBoWPath);
            try (BufferedWriter writer = new BufferedWriter(new FileWriter(testBoWPath))) {
                writer.write(testBoW.toString());
            }

            // 4. Verificar headers
            System.out.println("Headers egiaztatzen...");
            ConverterUtils.DataSource trainBoWSource = new ConverterUtils.DataSource(trainBoWPath);
            Instances trainBoW = trainBoWSource.getDataSet();

            if (trainBoW.numAttributes() != testBoW.numAttributes()) {
                System.out.println("Abisua: Train eta test multzoen atributu kopurua ez da berdina.");
            } else {
                for (int i = 0; i < trainBoW.numAttributes(); i++) {
                    String trainAttrName = trainBoW.attribute(i).name();
                    String testAttrName = testBoW.attribute(i).name();
                    if (!trainAttrName.equals(testAttrName)) {
                        System.out.println("Abisua: Atributuen izenak ez datoz bat: " + trainAttrName + " vs " + testAttrName);
                    }
                }
            }

            System.out.println("Prozesua amaitu da.");
        } catch (Exception e) {
            e.printStackTrace();
            System.out.println("Errorea prozesuan.");
        }
    }

}
