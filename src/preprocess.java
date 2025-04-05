import com.opencsv.CSVReader;
import com.opencsv.exceptions.CsvException;

import java.io.*;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.*;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.converters.ArffSaver;
import weka.core.tokenizers.WordTokenizer;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.FixedDictionaryStringToWordVector;
import weka.filters.unsupervised.attribute.Reorder;
import weka.filters.unsupervised.attribute.StringToWordVector;
import weka.filters.unsupervised.instance.Randomize;
import weka.filters.unsupervised.instance.RemovePercentage;

public class preprocess {
    public static void main(String[] args) throws IOException {
        // Argumentuak ondo pasatu diren konprobatu
        if (args.length != 10) {
            System.out.println("Erabilera: java preproces <train_csv> <test_csv> <train_RAW.arff> <test_RAW.arff> <train_Split.arff> <dev_Split.arff> <hiztegia.txt> <train_split_BOW.arff> <dev_split_BOW.arff> <test_BOW.arff>" );
            return;
        }
        String inTrainCSVPath = args[0]; // (in) CSV fitxategiaren helbidea, train
        String inTestCSVPath = args[1]; // (in) CSV fitxategiaren helbidea, test
        String outTrainARFFPath = args[2]; // (out) ARFF fitxategiaren helbide, train
        String outTestARFFPath = args[3]; // (out) ARFF fitxategiaren helbide, test
        String outTrainSplitPath = args[4];
        String outDevSplitPath = args[5];
        String outDictionaryPath = args[6];
        String outTrainBOWPath = args[7];
        String outDevBOWPath = args[8];
        String outTestBOWPath = args[9];

        Map<String, String> labelMapping = loadLabelMapping();

        try {

            //getARFF
            // 1. train prozesatu erabilitako mailak eta tokiak lortzeko
            Set<String> trainCategories = new HashSet<>();
            Set<String> places = new HashSet<>();
            processCSV(inTrainCSVPath, labelMapping, false, trainCategories, places);

            // 2. Mapeatutako maila guztiak barne daudela ziurtatu
            trainCategories.addAll(labelMapping.values());

            // 3. Fitxategiak bihurtu maila berdinak erabiliz
            convertARFF(inTrainCSVPath, outTrainARFFPath, labelMapping, false, places, trainCategories);
            convertARFF(inTestCSVPath, outTestARFFPath, labelMapping, true, places, trainCategories);

            System.out.println("ARFF fitxategiak zuzen sortu dira.");

            // getSplit

            DataSource source = new DataSource(outTrainARFFPath);
            Instances data = source.getDataSet();
            data.setClassIndex(data.numAttributes() - 1);

            Randomize randomFilter = new Randomize();
            randomFilter.setRandomSeed(1);
            randomFilter.setInputFormat(data);
            Instances randomData = Filter.useFilter(data, randomFilter);

            // 2. Primero obtener TRAIN (80%)
            RemovePercentage removeFilter = new RemovePercentage();
            removeFilter.setInputFormat(data);
            removeFilter.setPercentage(80); // Eliminar 20%
            removeFilter.setInvertSelection(true); // Conservar 80%
            Instances trainData = Filter.useFilter(randomData, removeFilter);

            // 3. Luego obtener DEV (20%) del original
            removeFilter = new RemovePercentage(); // Reiniciar filtro
            removeFilter.setInputFormat(data);
            removeFilter.setPercentage(80); // Eliminar 80%
            removeFilter.setInvertSelection(false); // Conservar 20%
            Instances devData = Filter.useFilter(randomData, removeFilter);

            // 5. Guardar
            ArffSaver saver = new ArffSaver();
            saver.setInstances(trainData);
            saver.setFile(new File(outTrainSplitPath));
            saver.writeBatch();

            saver.setInstances(devData);
            saver.setFile(new File(outDevSplitPath));
            saver.writeBatch();

            //arff2BoW
            Instances trainBOWData = processTrainingData(outTrainSplitPath, outDictionaryPath, outTrainBOWPath);

            processTestData(outDevSplitPath, outDevBOWPath, outDictionaryPath, trainBOWData);

            processTestData(outTestARFFPath, outTestBOWPath, outDictionaryPath, trainBOWData);

            System.out.println("Prozesua zuzenki burutu da.");

        }
        catch (Exception e) {
            e.printStackTrace();
            System.out.println("ERROR");
        }
    }

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

    // header idazteko metodoa
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

    // narrative garbitzeko metodoa
    private static String cleanNarrative(String text) {
        return text.replace("#", "").replace("?", "").replace("!", "").replace(",", "")
                .replace("\"", "").replace("'", "").replace(";", "").replace(":", "")
                .replace("(", "").replace(")", "").replace("[", "").replace("]", "")
                .replace("{", "").replace("}", "").replace("/", " ").replace("\\", " ")
                .trim();
    }

    // Hiltze arrazoien multzokatzea kargatu
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

    // arff2BoW
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

    private static Instances moveClassToLast(Instances data) throws Exception {
        if (data.classIndex() == -1) {
            throw new IllegalStateException("Klasea ez dagi zehaztuta.");
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
