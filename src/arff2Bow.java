import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;
import weka.filters.unsupervised.attribute.FixedDictionaryStringToWordVector;
import weka.filters.unsupervised.attribute.Reorder;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;

public class arff2Bow {
    public static void main(String[] args) throws Exception {
        // Verificar que se han pasado los argumentos correctamente
        if (args.length != 7) {
            System.out.println("Erabilera: java arffToBow <train_split_RAW.arff> <dev_split_RAW.arff> <test_RAW.arff> <hiztegia.txt> <train_split_BOW.arff> <dev_split_BOW.arff> <test_BOW.arff>");
            return;
        }

        try {
            // Asignar los parámetros de entrada
            String inTrainRAWPath = args[0];      // Ruta al archivo train_split_RAW.arff
            String inDevRAWPath = args[1];        // Ruta al archivo dev_split_RAW.arff
            String inTestRAWPath = args[2];       // Ruta al archivo test_RAW.arff
            String outDictionaryPath = args[3];   // Ruta al archivo dictionary.txt
            String outTrainBOWPath = args[4];     // Ruta para guardar train_split_BOW.arff
            String outDevBOWPath = args[5];       // Ruta para guardar dev_split_BOW.arff
            String outTestBOWPath = args[6];      // Ruta para guardar test_BOW.arff

            // 1. Procesar el archivo de entrenamiento (generar diccionario y BoW)
            System.out.println("Procesando train_split_RAW.arff...");
            Instances trainBOWData = processTrainingData(inTrainRAWPath, outDictionaryPath, outTrainBOWPath);

            // 2. Procesar el archivo de desarrollo (usar diccionario y verificar headers)
            System.out.println("Procesando dev_split_RAW.arff...");
            processTestData(inDevRAWPath, outDevBOWPath, outDictionaryPath, trainBOWData);

            // 3. Procesar el archivo de prueba (usar diccionario y verificar headers)
            System.out.println("Procesando test_RAW.arff...");
            processTestData(inTestRAWPath, outTestBOWPath, outDictionaryPath, trainBOWData);

            System.out.println("Prozesua amaitu da.");
        } catch (Exception e) {
            e.printStackTrace();
            System.out.println("ERROR");
        }
    }

    // Método para procesar el archivo de entrenamiento
    private static Instances processTrainingData(String trainRAWPath, String dictionaryPath, String trainBOWPath) throws Exception {
        // Cargar el archivo RAW
        DataSource source = new DataSource(trainRAWPath);
        Instances trainRAWData = source.getDataSet();

        // Definir la clase si no está definida
        if (trainRAWData.classIndex() == -1) {
            trainRAWData.setClassIndex(trainRAWData.attribute("Cause_of_Death").index());
        }

        // Aplicar StringToWordVector para generar el diccionario y BoW
        StringToWordVector filter = new StringToWordVector();
        filter.setOutputWordCounts(false);  // No contar la frecuencia de palabras
        filter.setLowerCaseTokens(true);    // Convertir a minúsculas
        filter.setDictionaryFileToSaveTo(new File(dictionaryPath)); // Guardar diccionario
        filter.setInputFormat(trainRAWData); // Establecer formato de entrada
        Instances trainBOWData = Filter.useFilter(trainRAWData, filter);

        // Mover la clase al último atributo
        trainBOWData = moveClassToLast(trainBOWData);

        // Guardar el archivo BOW
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(trainBOWPath))) {
            writer.write(trainBOWData.toString());
        }

        return trainBOWData; // Devolver el conjunto de datos transformado
    }

    // Método para procesar archivos de prueba o desarrollo
    private static void processTestData(String dev_testRAWPath, String bowPath, String dictionaryPath, Instances trainBOWData) throws Exception {
        // Cargar el archivo RAW
        DataSource source = new DataSource(dev_testRAWPath);
        Instances dev_testRAWData = source.getDataSet();

        // Definir la clase si no está definida
        if (dev_testRAWData.classIndex() == -1) {
            dev_testRAWData.setClassIndex(dev_testRAWData.attribute("Cause_of_Death").index());
        }

        // Aplicar FixedDictionaryStringToWordVector usando el diccionario
        FixedDictionaryStringToWordVector filter = new FixedDictionaryStringToWordVector();
        filter.setDictionaryFile(new File(dictionaryPath)); // Cargar diccionario
        filter.setOutputWordCounts(false);  // No contar la frecuencia de palabras
        filter.setInputFormat(dev_testRAWData); // Establecer formato de entrada
        Instances bowData = Filter.useFilter(dev_testRAWData, filter);

        // Mover la clase al último atributo
        bowData = moveClassToLast(bowData);

        // Verificar headers con el archivo de entrenamiento
        System.out.println("Headers egiaztatzen...");
        if (trainBOWData.numAttributes() != bowData.numAttributes()) {
            System.out.println("Abisua: Train eta test multzoen atributu kopurua ez da berdina.");
        } else {
            for (int i = 0; i < trainBOWData.numAttributes(); i++) {
                String trainAttrName = trainBOWData.attribute(i).name();
                String testAttrName = bowData.attribute(i).name();
                if (!trainAttrName.equals(testAttrName)) {
                    System.out.println("Abisua: Atributuen izenak ez datoz bat: " + trainAttrName + " vs " + testAttrName);
                }
            }
        }

        // Guardar el archivo BOW
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(bowPath))) {
            writer.write(bowData.toString());
        }
    }

    // Método para mover la clase al último atributo
    private static Instances moveClassToLast(Instances data) throws Exception {
        if (data.classIndex() == -1) {
            throw new IllegalStateException("La clase no está definida.");
        }

        // Crear una cadena de índices para mover la clase al final
        StringBuilder indices = new StringBuilder();
        for (int i = 0; i < data.numAttributes(); i++) {
            if (i != data.classIndex()) {
                indices.append(i + 1).append(","); // +1 porque Weka usa índices basados en 1
            }
        }
        indices.append(data.classIndex() + 1); // Mover la clase al final

        // Aplicar el filtro Reorder
        Reorder reorderFilter = new Reorder();
        reorderFilter.setAttributeIndices(indices.toString());
        reorderFilter.setInputFormat(data);
        return Filter.useFilter(data, reorderFilter);
    }
}
