import com.opencsv.CSVReader;
import com.opencsv.exceptions.CsvException;

import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

public class getARFF {
    public static void main(String[] args) throws IOException {
        // Verificar que se han pasado los argumentos correctamente
        if (args.length != 4) {
            System.out.println("Erabilera: java getArff <train_csv> <test_csv> <train_RAW.arff> <test_RAW.arff>");
            return;
        }
        String inTrainCSVPath = args[0]; // Ruta al archivo CSV de entrenamiento
        String inTestCSVPath = args[1]; // Ruta al archivo CSV de prueba
        String outTrainARFFPath = args[2]; // Ruta al archivo ARFF de salida de entrenamiento
        String outTestARFFPath = args[3]; // Ruta al archivo ARFF de salida de prueba

        Map<String, String> labelMapping = loadLabelMapping();

        try {
            // 1. Procesar train para obtener categorías y lugares usados
            Set<String> trainCategories = new HashSet<>();
            Set<String> places = new HashSet<>();
            processCSV(inTrainCSVPath, labelMapping, false, trainCategories, places);

            // 2. Asegurar que todas las categorías del mapeo estén incluidas
            trainCategories.addAll(labelMapping.values());

            // 3. Convertir archivos usando las mismas categorías
            convertARFF(inTrainCSVPath, outTrainARFFPath, labelMapping, false, places, trainCategories);
            convertARFF(inTestCSVPath, outTestARFFPath, labelMapping, true, places, trainCategories);

            System.out.println("Archivos ARFF generados exitosamente.");
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

        // Escribir cabecera usando tu método
        writeHeaderARFF(writer, places, categories);

        // Procesar y escribir datos
        List<String[]> data = reader.readAll();
        for (int i = 1; i < data.size(); i++) {
            String[] row = data.get(i);

            // Procesar Cause_of_Death
            if (isTest || row[6].equalsIgnoreCase("na")) {
                row[6] = "?";
            } else {
                row[6] = labelMapping.getOrDefault(row[6].trim().toLowerCase(), row[6])
                        .replaceAll("[^a-zA-Z0-9]", "_");
            }

            // Limpiar narrative
            row[5] = "\"" + cleanNarrative(row[5]) + "\"";

            writer.write(String.join(",", row) + "\n");
        }

        reader.close();
        writer.close();
    }

    // Escribir la cabecera
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

    // Limpiar el texto de Narrative
    private static String cleanNarrative(String text) {
        return text.replace("#", "").replace("?", "").replace("!", "").replace(",", "")
                .replace("\"", "").replace("'", "").replace(";", "").replace(":", "")
                .replace("(", "").replace(")", "").replace("[", "").replace("]", "")
                .replace("{", "").replace("}", "").replace("/", " ").replace("\\", " ")
                .trim();
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
}
