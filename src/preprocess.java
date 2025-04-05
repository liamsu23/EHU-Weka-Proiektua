import com.opencsv.CSVReader;
import com.opencsv.exceptions.CsvException;

import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.converters.ArffSaver;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.instance.Randomize;
import weka.filters.unsupervised.instance.RemovePercentage;
import weka.filters.supervised.instance.Resample;
import weka.core.tokenizers.WordTokenizer;
import weka.filters.unsupervised.attribute.StringToWordVector;
import weka.filters.unsupervised.attribute.FixedDictionaryStringToWordVector;
import weka.filters.unsupervised.attribute.Reorder;

import java.io.*;
import java.util.*;


public class preprocess {
    public static void main(String[] args) throws IOException {
        // Argumentuak ondo pasatu diren konprobatu
        if (args.length != 13) {
            System.out.println("Erabilera: java preprocess <train_csv> <test_csv> <train_RAW.arff> <test_RAW.arff> <train_split_RAW.arff> <dev_split_RAW.arff> <dictionary.txt> <train_split_BOW.arff> <dev_split_BOW.arff> <test_BOW.arff> <train_split_BOW_FSS.arff> <dev_split_BOW_FSS.arff> <test_BOW_FSS.arff>");
            return;
        }
        // Argumentos para getARFF
        String inTrainCSVPath = args[0];
        String inTestCSVPath = args[1];
        String outTrainARFFPath = args[2];
        String outTestARFFPath = args[3];

        // Argumentos para getSplit
        //String inTrainPath = args[0]; el mismo que args[2]
        String outTrainSplitPath = args[4];
        String outDevSplitPath = args[5];

        // Argumentos para arff2bow
        //String inTrainRAWPath = args[0]; el mismo que args[4]
        //String inDevRAWPath = args[1]; el mismo que args[5]
        //String inTestRAWPath = args[2]; el mismo que args[3]
        String outDictionaryPath = args[6];
        String outTrainBOWPath = args[7];
        String outDevBOWPath = args[8];
        String outTestBOWPath = args[9];

        // Argumentos para fssInfoGain
        //String inTrainBOWPath = args[0]; el mismo que args[7]
        //String inDevBOWPath = args[1]; el mismo que args[8]
        //String inTestBOWPath = args[2]; el mismo que args[9]
        String outTrainBOWFSSPath = args[10];
        String outDevBOWFSSPath = args[11];
        String outTestBOWFSSPath = args[12];

        Map<String, String> labelMapping = loadLabelMapping();

        try {
            // === PASO 1: Convertir CSV a ARFF (getARFF) ===
            System.out.println("=== PASO 1/4: Convirtiendo CSV a ARFF ===");
            Set<String> trainCategories = new HashSet<>();
            Set<String> places = new HashSet<>();

            // Procesar train para obtener categorías y lugares
            processCSV(inTrainCSVPath, labelMapping, false, trainCategories, places);
            trainCategories.addAll(labelMapping.values());

            // Convertir a ARFF
            convertARFF(inTrainCSVPath, outTrainARFFPath, labelMapping, false, places, trainCategories);
            convertARFF(inTestCSVPath, outTestARFFPath, labelMapping, true, places, trainCategories);

            // === PASO 2: Dividir datos (getSplit) ===
            System.out.println("\n=== PASO 2/4: Dividiendo dataset de entrenamiento ===");
            splitData(outTrainARFFPath, outTrainSplitPath, outDevSplitPath);

            // === PASO 3: Bag-of-Words (arff2bow) ===
            System.out.println("\n=== PASO 3/4: Creando representación Bag-of-Words ===");
            processBOW(outTrainSplitPath, outDevSplitPath, outTestARFFPath, outDictionaryPath, outTrainBOWPath, outDevBOWPath, outTestBOWPath);

            // === PASO 4: Selección de características (fssInfoGain) ===
            System.out.println("\n=== PASO 4/4: Selección de características con InfoGain ===");
            featureSelection(outTrainBOWPath, outDevBOWPath, outTestBOWPath, outTrainBOWFSSPath, outDevBOWFSSPath, outTestBOWFSSPath);

            System.out.println("\n¡Proceso completado con éxito!");
        } catch (Exception e) {
            e.printStackTrace();
            System.out.println("ERROR durante el procesamiento");
        }
    }

    // ============== getARFF Methods ==============
    private static void processCSV(String csvFile, Map<String, String> labelMapping, boolean isTest, Set<String> categories, Set<String> places) throws IOException, CsvException {
        CSVReader reader = new CSVReader(new FileReader(csvFile));
        List<String[]> data = reader.readAll();

        for (int i = 1; i < data.size(); i++) {
            String[] row = data.get(i);
            places.add(row[4]); // Site

            if (!isTest && !row[6].equalsIgnoreCase("na")) {
                String mappedLabel = labelMapping.getOrDefault(row[6].trim().toLowerCase(), row[6])
                        .replaceAll("[^a-zA-Z0-9]", "_");
                categories.add(mappedLabel);
            }
        }
        reader.close();
    }

    private static void convertARFF(String csvFile, String arffFile, Map<String, String> labelMapping, boolean isTest, Set<String> places, Set<String> categories) throws IOException, CsvException {
        CSVReader reader = new CSVReader(new FileReader(csvFile));
        FileWriter writer = new FileWriter(arffFile);

        // header idatzi
        writeHeaderARFF(writer, places, categories);

        // Datuak prozesatu eta idatzi
        List<String[]> data = reader.readAll();
        for (int i = 1; i < data.size(); i++) {
            String[] row = data.get(i);

            // Cause_of_Death prozesatu
            if (isTest || row[6].equalsIgnoreCase("na")) {
                row[6] = "?";
            } else {
                row[6] = labelMapping.getOrDefault(row[6].trim().toLowerCase(), row[6])
                        .replaceAll("[^a-zA-Z0-9]", "_");
            }

            // narrative garbitu
            row[5] = "\"" + cleanNarrative(row[5]) + "\"";

            writer.write(String.join(",", row) + "\n");
        }

        reader.close();
        writer.close();
    }

    private static void writeHeaderARFF(FileWriter fw, Set<String> places, Set<String> generalCategories) throws IOException {
        try {
            fw.write("@RELATION muertes_causas\n\n");
            fw.write("@ATTRIBUTE ID NUMERIC\n");
            fw.write("@ATTRIBUTE Module {Adult, Child, Neonate}\n");
            fw.write("@ATTRIBUTE Age NUMERIC\n");
            fw.write("@ATTRIBUTE Sex {1, 2}\n");
            fw.write("@ATTRIBUTE Site {" + String.join(",", places) + "}\n");
            fw.write("@ATTRIBUTE Open_Response STRING\n");
            fw.write("@ATTRIBUTE Cause_of_Death {" + String.join(",", generalCategories) + "}\n\n");
            fw.write("@DATA\n");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static String cleanNarrative(String text) {
        return text.replace("#", "").replace("?", "").replace("!", "").replace(",", "")
                .replace("\"", "").replace("'", "").replace(";", "").replace(":", "")
                .replace("(", "").replace(")", "").replace("[", "").replace("]", "")
                .replace("{", "").replace("}", "").replace("/", " ").replace("\\", " ")
                .trim();
    }

    private static Map<String, String> loadLabelMapping() {
        Map<String, String> labelMapping = new HashMap<>();

        // Gaixotasun infekziosoak (CIE-10: A00-B99)
        labelMapping.put("diarrhea/dysentery", "Certain_infectious_and_Parasitic_Diseases");
        labelMapping.put("other infectious diseases", "Certain_infectious_and_Parasitic_Diseases");
        labelMapping.put("aids", "Certain_infectious_and_Parasitic_Diseases");
        labelMapping.put("sepsis", "Certain_infectious_and_Parasitic_Diseases");
        labelMapping.put("meningitis", "Certain_infectious_and_Parasitic_Diseases");
        labelMapping.put("meningitis/sepsis", "Certain_infectious_and_Parasitic_Diseases");
        labelMapping.put("malaria", "Certain_infectious_and_Parasitic_Diseases");
        labelMapping.put("encephalitis", "Certain_infectious_and_Parasitic_Diseases");
        labelMapping.put("measles", "Certain_infectious_and_Parasitic_Diseases");
        labelMapping.put("hemorrhagic fever", "Certain_infectious_and_Parasitic_Diseases");
        labelMapping.put("tb", "Certain_infectious_and_Parasitic_Diseases");
        labelMapping.put("hepatitis", "Certain_infectious_and_Parasitic_Diseases");
        labelMapping.put("tetanus", "Certain_infectious_and_Parasitic_Diseases");

        // Tumoreak (CIE-10: C00-D49)
        labelMapping.put("leukemia/lymphomas", "Neoplasms");
        labelMapping.put("colorectal cancer", "Neoplasms");
        labelMapping.put("lung cancer", "Neoplasms");
        labelMapping.put("cervical cancer", "Neoplasms");
        labelMapping.put("breast cancer", "Neoplasms");
        labelMapping.put("stomach cancer", "Neoplasms");
        labelMapping.put("prostate cancer", "Neoplasms");
        labelMapping.put("esophageal cancer", "Neoplasms");
        labelMapping.put("liver cancer", "Neoplasms");
        labelMapping.put("pancreatic cancer", "Neoplasms");
        labelMapping.put("other cancers", "Neoplasms");

        // Gaixotasun endokrinoak (CIE-10: E00-E90)
        labelMapping.put("diabetes", "Endocrine_Nutritional_and_Metabolic_Diseases");
        labelMapping.put("malnutrition", "Endocrine_Nutritional_and_Metabolic_Diseases");
        labelMapping.put("obesity", "Endocrine_Nutritional_and_Metabolic_Diseases");

        // Gaixotasun neurologikoak (CIE-10: G00-G99)
        labelMapping.put("epilepsy", "Diseases_of_the_Nervous_System");
        labelMapping.put("alzheimer", "Diseases_of_the_Nervous_System");
        labelMapping.put("parkinson", "Diseases_of_the_Nervous_System");

        // Gaixotasun kardiobaskularrak (CIE-10: I00-I99)
        labelMapping.put("stroke", "Diseases_of_the_circulatory_system");
        labelMapping.put("acute myocardial infarction", "Diseases_of_the_circulatory_system");
        labelMapping.put("heart failure", "Diseases_of_the_circulatory_system");
        labelMapping.put("hypertension", "Diseases_of_the_circulatory_system");
        labelMapping.put("cardiac arrest", "Diseases_of_the_circulatory_system");

        // Arnasketa-gaixotasunak (CIE-10: J00-J99)
        labelMapping.put("pneumonia", "Diseases_of_Respiratory_System");
        labelMapping.put("asthma", "Diseases_of_Respiratory_System");
        labelMapping.put("copd", "Diseases_of_Respiratory_System");
        labelMapping.put("tuberculosis respiratory", "Diseases_of_Respiratory_System");

        // Digestio-gaixotasunak (CIE-10: K00-K95)
        labelMapping.put("cirrhosis", "Diseases_of_the_Digestive_System");
        labelMapping.put("other digestive diseases", "Diseases_of_the_Digestive_System");
        labelMapping.put("gastritis", "Diseases_of_the_Digestive_System");
        labelMapping.put("peptic ulcer", "Diseases_of_the_Digestive_System");

        // Gaixotasun genitourinarioak (CIE-10: N00-N99)
        labelMapping.put("renal failure", "Diseases_of_the_Genitourinary_System");
        labelMapping.put("kidney disease", "Diseases_of_the_Genitourinary_System");

        // Haurdunaldi - Erditze (CIE-10: O00-O9A)
        labelMapping.put("preterm delivery", "Pregnancy_childbirth_and_the_puerperium");
        labelMapping.put("stillbirth", "Pregnancy_childbirth_and_the_puerperium");
        labelMapping.put("maternal", "Pregnancy_childbirth_and_the_puerperium");
        labelMapping.put("birth asphyxia", "Pregnancy_childbirth_and_the_puerperium");
        labelMapping.put("postpartum hemorrhage", "Pregnancy_childbirth_and_the_puerperium");

        // Sortzetiko malformazioak (CIE-10: Q00-Q99)
        labelMapping.put("congenital malformation", "Congenital_Malformations");

        // Kanpo-arrazoiak (CIE-10: V01-Y99)
        labelMapping.put("bite of venomous animal", "Injury_Poisoning_and_External_Causes");
        labelMapping.put("poisonings", "Injury_Poisoning_and_External_Causes");
        labelMapping.put("road traffic", "External_Causes_of_Morbidity_and_Mortality");
        labelMapping.put("falls", "External_Causes_of_Morbidity_and_Mortality");
        labelMapping.put("homicide", "External_Causes_of_Morbidity_and_Mortality");
        labelMapping.put("fires", "External_Causes_of_Morbidity_and_Mortality");
        labelMapping.put("drowning", "External_Causes_of_Morbidity_and_Mortality");
        labelMapping.put("suicide", "External_Causes_of_Morbidity_and_Mortality");
        labelMapping.put("violent death", "External_Causes_of_Morbidity_and_Mortality");
        labelMapping.put("other injuries", "External_Causes_of_Morbidity_and_Mortality");

        return labelMapping;
    }

    // ============== getSplit Methods ==============
    private static void splitData(String inputARFF, String outTrain, String outDev) throws Exception {
        DataSource source = new DataSource(inputARFF);
        Instances data = source.getDataSet();
        data.setClassIndex(data.numAttributes() - 1);

        // Balancear clases
        Resample resample = new Resample();
        resample.setBiasToUniformClass(1.0);
        resample.setInputFormat(data);
        Instances balancedData = Filter.useFilter(data, resample);

        // Aleatorizar
        Randomize randomize = new Randomize();
        randomize.setInputFormat(balancedData);
        Instances randomizedData = Filter.useFilter(balancedData, randomize);

        // Split 80/20
        RemovePercentage splitFilter = new RemovePercentage();
        splitFilter.setPercentage(80);

        splitFilter.setInvertSelection(true);
        splitFilter.setInputFormat(randomizedData);
        Instances trainData = Filter.useFilter(randomizedData, splitFilter);

        splitFilter.setInvertSelection(false);
        Instances devData = Filter.useFilter(randomizedData, splitFilter);

        saveARFF(trainData, outTrain);
        saveARFF(devData, outDev);
    }

    // ============== arff2bow Methods ==============
    private static void processBOW(String trainARFF, String devARFF, String testARFF, String dictPath, String outTrain, String outDev, String outTest) throws Exception {
        System.out.println("Procesando train_split_RAW.arff...");
        Instances trainBOWData = processTrainingData(trainARFF, dictPath, outTrain);

        System.out.println("Procesando dev_split_RAW.arff...");
        processTestData(devARFF, outDev, dictPath, trainBOWData);

        System.out.println("Procesando test_RAW.arff...");
        processTestData(testARFF, outTest, dictPath, trainBOWData);

        System.out.println("Proceso BOW finalizado con éxito.");
    }

    private static Instances processTrainingData(String trainRAWPath, String dictionaryPath, String trainBOWPath) throws Exception {
        DataSource source = new DataSource(trainRAWPath);
        Instances trainRAWData = source.getDataSet();

        if (trainRAWData.classIndex() == -1) {
            trainRAWData.setClassIndex(trainRAWData.attribute("Cause_of_Death").index());
        }

        // Configuración del filtro StringToWordVector
        StringToWordVector filter = new StringToWordVector();
        filter.setOutputWordCounts(true);
        filter.setLowerCaseTokens(true);
        filter.setDictionaryFileToSaveTo(new File(dictionaryPath));

        // Configurar tokenizador para eliminar caracteres especiales
        WordTokenizer tokenizer = new WordTokenizer();
        tokenizer.setDelimiters(" \r\n\t.,;:'\"()?![]@%&*/+-=<>{}^|#");
        filter.setTokenizer(tokenizer);

        filter.setInputFormat(trainRAWData);
        Instances trainBOWData = Filter.useFilter(trainRAWData, filter);

        trainBOWData = moveClassToLast(trainBOWData);

        try (BufferedWriter writer = new BufferedWriter(new FileWriter(trainBOWPath))) {
            writer.write(trainBOWData.toString());
        }

        return trainBOWData;
    }

    private static void processTestData(String dev_testRAWPath, String bowPath, String dictionaryPath, Instances trainBOWData) throws Exception {
        DataSource source = new DataSource(dev_testRAWPath);
        Instances dev_testRAWData = source.getDataSet();

        if (dev_testRAWData.classIndex() == -1) {
            dev_testRAWData.setClassIndex(dev_testRAWData.attribute("Cause_of_Death").index());
        }

        // Configurar FixedDictionaryStringToWordVector con el mismo tokenizador
        FixedDictionaryStringToWordVector filter = new FixedDictionaryStringToWordVector();
        filter.setDictionaryFile(new File(dictionaryPath));
        filter.setOutputWordCounts(true);

        WordTokenizer tokenizer = new WordTokenizer();
        tokenizer.setDelimiters(" \r\n\t.,;:'\"()?![]@%&*/+-=<>{}^|#");
        filter.setTokenizer(tokenizer);

        filter.setInputFormat(dev_testRAWData);
        Instances bowData = Filter.useFilter(dev_testRAWData, filter);

        bowData = moveClassToLast(bowData);

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

        try (BufferedWriter writer = new BufferedWriter(new FileWriter(bowPath))) {
            writer.write(bowData.toString());
        }
    }

    // ============== fssInfoGain Methods ==============
    private static void featureSelection(String trainBOW, String devBOW, String testBOW, String outTrain, String outDev, String outTest) throws Exception {
        System.out.println("Procesando train_split_BOW.arff...");
        Instances trainBOWFSSData = processTrainingDataFSS(trainBOW, outTrain);

        System.out.println("Procesando dev_split_BOW.arff...");
        processTestDataFSS(devBOW, outDev, trainBOWFSSData);

        System.out.println("Procesando test_BOW.arff...");
        processTestDataFSS(testBOW, outTest, trainBOWFSSData);

        System.out.println("Proceso de selección de características completado.");
    }

    private static Instances processTrainingDataFSS(String inTrainPath, String outTrainPath) throws Exception {
        // Cargar datos de entrenamiento
        DataSource source = new DataSource(inTrainPath);
        Instances data = source.getDataSet();

        // Establecer atributo clase
        if (data.classIndex() == -1) {
            data.setClassIndex(data.attribute("Cause_of_Death").index());
        }

        System.out.println("Número de atributos antes de la selección: " + data.numAttributes());

        // Configurar y aplicar filtro de selección de atributos
        AttributeSelection filter = createFeatureSelector();
        filter.setInputFormat(data);
        Instances selectedData = Filter.useFilter(data, filter);

        System.out.println("Número de atributos después de la selección: " + selectedData.numAttributes());

        // Guardar datos procesados
        saveARFF(selectedData, outTrainPath);

        return selectedData;
    }

    private static void processTestDataFSS(String inTestPath, String outTestPath, Instances trainData) throws Exception {
        // Cargar datos de test/dev
        DataSource source = new DataSource(inTestPath);
        Instances testData = source.getDataSet();

        // Asegurar que el atributo clase está establecido (si existe)
        if (testData.attribute("Cause_of_Death") != null) {
            testData.setClassIndex(testData.attribute("Cause_of_Death").index());
        }

        // Ajustar datos de test para que coincidan con el formato de entrenamiento
        Instances adjustedData = adjustHeaders(testData, trainData);

        // Guardar datos procesados
        saveARFF(adjustedData, outTestPath);
    }

    private static AttributeSelection createFeatureSelector() {
        AttributeSelection filter = new AttributeSelection();
        InfoGainAttributeEval evaluator = new InfoGainAttributeEval();
        Ranker search = new Ranker();
        search.setNumToSelect(1000); // Número de atributos a seleccionar

        filter.setEvaluator(evaluator);
        filter.setSearch(search);

        return filter;
    }

    private static Instances adjustHeaders(Instances testData, Instances trainData) throws Exception {
        // 1. Eliminar atributos que están en test pero no en train
        ArrayList<Integer> indicesToRemove = new ArrayList<>();
        for (int i = 0; i < testData.numAttributes(); i++) {
            if (trainData.attribute(testData.attribute(i).name()) == null) {
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
        for (int i = 0; i < trainData.numAttributes(); i++) {
            int idx = testData.attribute(trainData.attribute(i).name()).index() + 1;
            order.append(idx).append(",");
        }
        order.deleteCharAt(order.length() - 1); // Eliminar última coma

        Reorder reorderFilter = new Reorder();
        reorderFilter.setAttributeIndices(order.toString());
        reorderFilter.setInputFormat(testData);
        testData = Filter.useFilter(testData, reorderFilter);

        return testData;
    }

    // ============== Utility Methods ==============
    private static Instances moveClassToLast(Instances data) throws Exception {
        if (data.classIndex() == -1) {
            throw new IllegalStateException("Class attribute not set");
        }

        StringBuilder indices = new StringBuilder();
        for (int i = 0; i < data.numAttributes(); i++) {
            if (i != data.classIndex()) {
                indices.append(i + 1).append(",");
            }
        }
        indices.append(data.classIndex() + 1);

        Reorder reorder = new Reorder();
        reorder.setAttributeIndices(indices.toString());
        reorder.setInputFormat(data);
        return Filter.useFilter(data, reorder);
    }

    private static void saveARFF(Instances data, String path) throws Exception {
        ArffSaver saver = new ArffSaver();
        saver.setInstances(data);
        saver.setFile(new File(path));
        saver.writeBatch();
    }
}
