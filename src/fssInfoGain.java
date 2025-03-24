import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.Reorder;

import java.io.File;
import java.io.FileOutputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;

public class fssInfoGain {
    public static void main(String[] args) throws Exception {
        // Verificar que se han pasado los argumentos correctamente
        if (args.length != 6) {
            System.out.println("Erabilera: java fssInfoGain <train_split_BOW.arff> <dev_split_BOW.arff> <test_BOW.arff> <train_split_BOW_FSS.arff> <dev_split_BOW_FSS.arff> <test_BOW_FSS.arff>");
            return;
        }

        try {
            // Asignar los parámetros de entrada
            String inTrainBOWPath = args[0];
            String inDevBOWPath = args[1];
            String inTestBOWPath = args[2];
            String outTrainBOWFSSPath = args[3];
            String outDevBOWFSSPath = args[4];
            String outTestBOWFSSPath = args[5];

            // 1. Procesar el archivo de entrenamiento
            System.out.println("Procesando train_split_BOW.arff...");
            Instances trainBOWFSSData = processTrainingData(inTrainBOWPath, outTrainBOWFSSPath);

            // 2. Procesar el archivo de desarrollo
            System.out.println("Procesando dev_split_BOW.arff...");
            processTestData(inDevBOWPath, outDevBOWFSSPath, trainBOWFSSData);

            // 3. Procesar el archivo de prueba
            System.out.println("Procesando test_BOW.arff...");
            processTestData(inTestBOWPath, outTestBOWFSSPath, trainBOWFSSData);

            System.out.println("Prozesua amaitu da.");
        } catch (Exception e) {
            e.printStackTrace();
            System.out.println("ERROR");
        }
    }

    /**
     * Procesa los datos de entrenamiento aplicando selección de características
     *
     * @param inputPath Ruta del archivo ARFF de entrada
     * @param outputPath Ruta donde guardar los datos procesados
     * @return Instancias procesadas
     * @throws Exception
     */
    private static Instances processTrainingData(String inputPath, String outputPath) throws Exception {
        // Cargar datos de entrenamiento
        DataSource source = new DataSource(inputPath);
        Instances data = source.getDataSet();

        // Establecer atributo clase (asumimos que se llama "Cause_of_Death")
        if (data.classIndex() == -1) {
            data.setClassIndex(data.attribute("Cause_of_Death").index());
        }

        System.out.println("📊 Número de atributos antes de la selección: " + data.numAttributes());

        // Configurar y aplicar filtro de selección de atributos
        AttributeSelection filter = createFeatureSelectionFilter();
        filter.setInputFormat(data);
        Instances selectedData = Filter.useFilter(data, filter);

        System.out.println("✅ Número de atributos después de la selección: " + selectedData.numAttributes());

        // Guardar datos procesados
        saveArffFile(selectedData, outputPath);

        return selectedData;
    }

    /**
     * Procesa datos de test/dev ajustándolos al formato de los datos de entrenamiento
     *
     * @param inputPath Ruta del archivo ARFF de entrada
     * @param outputPath Ruta donde guardar los datos procesados
     * @param trainDataTemplate Datos de entrenamiento procesados (como referencia)
     * @throws Exception
     */
    private static void processTestData(String inputPath, String outputPath, Instances trainDataTemplate) throws Exception {
        // Cargar datos de test/dev
        DataSource source = new DataSource(inputPath);
        Instances testData = source.getDataSet();

        // Asegurar que el atributo clase está establecido (si existe)
        if (testData.attribute("Cause_of_Death") != null) {
            testData.setClassIndex(testData.attribute("Cause_of_Death").index());
        }

        // Ajustar datos de test para que coincidan con el formato de entrenamiento
        Instances adjustedData = adjustTestDataToTrainFormat(testData, trainDataTemplate);

        // Guardar datos procesados
        saveArffFile(adjustedData, outputPath);
    }

    /**
     * Crea y configura el filtro para selección de características
     *
     * @return AttributeSelection configurado
     */
    private static AttributeSelection createFeatureSelectionFilter() {
        AttributeSelection filter = new AttributeSelection();
        InfoGainAttributeEval evaluator = new InfoGainAttributeEval();
        Ranker search = new Ranker();
        search.setNumToSelect(1000); // Número de atributos a seleccionar

        filter.setEvaluator(evaluator);
        filter.setSearch(search);

        return filter;
    }

    /**
     * Ajusta los datos de test para que coincidan con el formato de entrenamiento
     *
     * @param testData Datos a ajustar
     * @param trainDataTemplate Datos de entrenamiento como referencia
     * @return Instancias ajustadas
     * @throws Exception
     */
    private static Instances adjustTestDataToTrainFormat(Instances testData, Instances trainDataTemplate) throws Exception {
        // 1. Eliminar atributos que están en test pero no en train
        ArrayList<Integer> indicesToRemove = new ArrayList<>();
        for (int i = 0; i < testData.numAttributes(); i++) {
            if (trainDataTemplate.attribute(testData.attribute(i).name()) == null) {
                indicesToRemove.add(i);
            }
        }

        if (!indicesToRemove.isEmpty()) {
            Remove removeFilter = new Remove();
            removeFilter.setAttributeIndicesArray(indicesToRemove.stream().mapToInt(i -> i).toArray());
            removeFilter.setInvertSelection(false);
            removeFilter.setInputFormat(testData);
            testData = Filter.useFilter(testData, removeFilter);
        }

        // 2. Reordenar atributos para que coincidan con train
        StringBuilder order = new StringBuilder();
        for (int i = 0; i < trainDataTemplate.numAttributes(); i++) {
            int idx = testData.attribute(trainDataTemplate.attribute(i).name()).index() + 1;
            order.append(idx).append(",");
        }
        order.deleteCharAt(order.length() - 1); // Eliminar última coma

        Reorder reorderFilter = new Reorder();
        reorderFilter.setAttributeIndices(order.toString());
        reorderFilter.setInputFormat(testData);
        testData = Filter.useFilter(testData, reorderFilter);

        return testData;
    }

    /**
     * Guarda un conjunto de datos en formato ARFF
     *
     * @param data Instancias a guardar
     * @param outputPath Ruta de destino
     * @throws Exception
     */
    private static void saveArffFile(Instances data, String outputPath) throws Exception {
        ArffSaver saver = new ArffSaver();
        saver.setInstances(data);
        saver.setFile(new File(outputPath));
        saver.writeBatch();
    }

}
