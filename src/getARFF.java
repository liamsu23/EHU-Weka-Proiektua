import com.opencsv.CSVReader;
import com.opencsv.exceptions.CsvException;
import weka.core.Instances;
import weka.core.converters.ArffSaver;

import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

public class getARFF {
    public static void main(String[] args) throws IOException {
        // Verificar que se han pasado los argumentos correctamente
        if (args.length != 4) {
            System.out.println("Erabilera: java getArff2 <train_csv> <test_csv> <train_arff> <test_arff>");
            return;
        }
        String trainCSVPath = args[0]; // Ruta al archivo CSV de entrenamiento
        String testCSVPath = args[1]; // Ruta al archivo CSV de prueba
        String trainARFFPath = args[2]; // Ruta al archivo ARFF de salida de entrenamiento
        String testARFFPath = args[3]; // Ruta al archivo ARFF de salida de prueba

        // Cargar el mapeo de etiquetas
        Map<String, String> labelMapping = loadLabelMapping();

        try {
            // Convertir train.csv a ARFF (con clase conocida)
            convertCSVtoARFF(trainCSVPath, trainARFFPath, labelMapping, false);

            // Convertir test.csv a ARFF (con clase desconocida "?")
            convertCSVtoARFF(testCSVPath, testARFFPath, labelMapping, true);

            System.out.println("Archivos ARFF generados exitosamente.");
        } catch (IOException | CsvException e) {
            e.printStackTrace();
            System.out.println("ERROR");
        }
    }

    // Método para convertir CSV a ARFF
    private static void convertCSVtoARFF(String csvFilePath, String arffFilePath, Map<String, String> labelMapping, boolean isTestData) throws IOException, CsvException {
        // Leer el archivo CSV
        CSVReader reader = new CSVReader(new FileReader(csvFilePath));
        List<String[]> csvData = reader.readAll();

        // Conjuntos para valores únicos de Place y Cause_of_Death
        Set<String> places = new HashSet<>();
        Set<String> generalCategories = new HashSet<>();

        // Extraer valores únicos de Place y Cause_of_Death
        for (int i = 1; i < csvData.size(); i++) {
            places.add(csvData.get(i)[4]); // Columna Place
            if (!isTestData) { // Solo mapear causas de muerte si no es un conjunto de prueba
                String specificLabel = csvData.get(i)[6].trim().toLowerCase(); // Columna Cause_of_Death
                String generalLabel = labelMapping.getOrDefault(specificLabel, specificLabel); // Mapeo
                generalLabel = generalLabel.replaceAll("[^a-zA-Z0-9]", "_"); // Limpiar la etiqueta
                generalCategories.add(generalLabel); // Añadir al conjunto
            }
        }

        // Si es un conjunto de prueba, la clase es desconocida ("?")
        if (isTestData) {
            generalCategories.add("?"); // Añadir "?" como clase desconocida
        }

        // Crear el archivo ARFF
        FileWriter arffWriter = new FileWriter(arffFilePath);

        // Escribir la cabecera en el archivo ARFF
        writeHeaderARFF(arffWriter, places, generalCategories);

        // Escribir los datos en el ARFF
        for (int i = 1; i < csvData.size(); i++) {
            String[] row = csvData.get(i);
            String specificLabel = row[6].trim().toLowerCase(); // Columna Cause_of_Death

            // Si es un conjunto de prueba, la clase es desconocida ("?")
            if (isTestData) {
                row[6] = "?";
            } else {
                // Si la clase es "NA" o "na", la convertimos en "?"
                if (specificLabel.equalsIgnoreCase("na")) {
                    row[6] = "?";
                } else {
                    // Aplicar el mapeo si existe, si no, usar el mismo valor
                    String generalLabel = labelMapping.getOrDefault(specificLabel, specificLabel);
                    generalLabel = generalLabel.replaceAll("[^a-zA-Z0-9?]", "_"); // Limpiar la etiqueta
                    row[6] = generalLabel; // Actualizar la causa de muerte
                }
            }

            row[5] = "\"" + cleanNarrative(row[5]) + "\""; // Limpiar la narrativa

            // Escribir la fila en el ARFF
            arffWriter.write(String.join(",", row) + "\n");
        }

        // Cerrar los archivos
        arffWriter.close();
        reader.close();
    }

    // Método para guardar Instances en un archivo ARFF
    private static void saveInstancesToARFF(Instances instances, String fileName) throws IOException {
        // train edo test multzoarekin lan egingo dugun jakin
        String[] split = fileName.split("\\\\");
        String multzoa = split[split.length - 1];
        System.out.println(multzoa);

        // test multzoa bada, klasearen balioa missing jarri
        if (multzoa.equals("test_RAW.arff")) {
            for (int i = 0; i < instances.numInstances(); i++) {
                instances.instance(i).setClassMissing();
            }
        }

        // .arff artxiboan gorde
        ArffSaver arffSaver = new ArffSaver();
        arffSaver.setInstances(instances);
        arffSaver.setFile(new File(fileName));
        arffSaver.writeBatch();
    }

    // Cargar la agrupación de causas de muerte
    private static Map<String, String> loadLabelMapping() {
        Map<String, String> labelMapping = new HashMap<>();
        labelMapping.put("Diarrhea/Dysentery", "Certain_infectious_and_Parasitic_Diseases");
        labelMapping.put("Other Infectious Diseases", "Certain_infectious_and_Parasitic_Diseases");
        labelMapping.put("AIDS", "Certain_infectious_and_Parasitic_Diseases");
        labelMapping.put("Sepsis", "Certain_infectious_and_Parasitic_Diseases");
        labelMapping.put("Meningitis", "Certain_infectious_and_Parasitic_Diseases");
        labelMapping.put("Meningitis/Sepsis", "Certain_infectious_and_Parasitic_Diseases");
        labelMapping.put("Malaria", "Certain_infectious_and_Parasitic_Diseases");
        labelMapping.put("Encephalitis", "Certain_infectious_and_Parasitic_Diseases");
        labelMapping.put("Measles", "Certain_infectious_and_Parasitic_Diseases");
        labelMapping.put("Hemorrhagic Fever", "Certain_infectious_and_Parasitic_Diseases");
        labelMapping.put("TB", "Certain_infectious_and_Parasitic_Diseases");

        labelMapping.put("Leukemia/Lymphomas", "Neoplasms");
        labelMapping.put("Colorectal Cancer", "Neoplasms");
        labelMapping.put("Lung Cancer", "Neoplasms");
        labelMapping.put("Cervical Cancer", "Neoplasms");
        labelMapping.put("Breast Cancer", "Neoplasms");
        labelMapping.put("Stomach Cancer", "Neoplasms");
        labelMapping.put("Prostate Cancer", "Neoplasms");
        labelMapping.put("Esophageal Cancer", "Neoplasms");
        labelMapping.put("Other Cancers", "Neoplasms");

        labelMapping.put("Diabetes", "Endocrine_Nutritional_and_Metabolic_Diseases");

        labelMapping.put("Epilepsy", "Diseases_of_the_Nervous_System");

        labelMapping.put("Stroke", "Diseases_of_the_circulatory_system");
        labelMapping.put("Acute Myocardial Infarction", "Diseases_of_the_circulatory_system");

        labelMapping.put("Pneumonia", "Diseases_of_Respiratory_System");
        labelMapping.put("Asthma", "Diseases_of_Respiratory_System");
        labelMapping.put("COPD", "Diseases_of_Respiratory_System");

        labelMapping.put("Cirrhosis", "Diseases_of_the_Digestive_System");
        labelMapping.put("Other Digestive Diseases", "Diseases_of_the_Digestive_System");

        labelMapping.put("Renal Failure", "Diseases_of_the_Genitourinary_System");

        labelMapping.put("Preterm Delivery", "Pregnancy_childbirth_and_the_puerperium");
        labelMapping.put("Stillbirth", "Pregnancy_childbirth_and_the_puerperium");
        labelMapping.put("Maternal", "Pregnancy_childbirth_and_the_puerperium");
        labelMapping.put("Birth Asphyxia", "Pregnancy_childbirth_and_the_puerperium");

        labelMapping.put("Congenital Malformations", "Congenital_Malformations");

        labelMapping.put("Bite of Venomous Animal", "Injury_Poisoning_and_External_Causes");
        labelMapping.put("Poisonings", "Injury_Poisoning_and_External_Causes");

        labelMapping.put("Road Traffic", "External_Causes_of_Morbidity_and_Mortality");
        labelMapping.put("Falls", "External_Causes_of_Morbidity_and_Mortality");
        labelMapping.put("Homicide", "External_Causes_of_Morbidity_and_Mortality");
        labelMapping.put("Fires", "External_Causes_of_Morbidity_and_Mortality");
        labelMapping.put("Drowning", "External_Causes_of_Morbidity_and_Mortality");
        labelMapping.put("Suicide", "External_Causes_of_Morbidity_and_Mortality");
        labelMapping.put("Violent Death", "External_Causes_of_Morbidity_and_Mortality");
        labelMapping.put("Other Injuries", "External_Causes_of_Morbidity_and_Mortality");

        return labelMapping;
    }

    // Limpiar el texto de Narrative
    private static String cleanNarrative(String text) {
        return text.replace("#", "").replace("?", "").replace("!", "").replace(",", "")
                .replace("\"", "").replace("'", "").replace(";", "").replace(":", "")
                .replace("(", "").replace(")", "").replace("[", "").replace("]", "")
                .replace("{", "").replace("}", "").replace("/", " ").replace("\\", " ")
                .trim();
    }

    // Escribir la cabecera en el archivo ARFF
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
}