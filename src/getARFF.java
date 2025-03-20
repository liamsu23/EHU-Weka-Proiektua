import com.opencsv.CSVReader;
import com.opencsv.exceptions.CsvException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;
import java.util.HashMap;
import java.util.Map;

public class getARFF {

    public static void main(String[] args) throws IOException {
        // Cargar los datos
        String csvFilePath = "hasierako_datuak_CSV/cleaned_PHMRC_VAI_redacted_free_text.train.csv"; // Ruta del archivo CSV
        String arffFilePath = "data/train_Zikina.arff"; // Ruta del archivo ARFF de salida
        // Cargar el mapeo de etiquetas
        Map<String, String> labelMapping = loadLabelMapping(); // Nombre de la relación ARFF

        try {
            // Leer el archivo CSV
            CSVReader reader = new CSVReader(new FileReader(csvFilePath));
            List<String[]> csvData = reader.readAll();

            // Crear el archivo ARFF
            FileWriter arffWriter = new FileWriter(arffFilePath);

            // Conjuntos para valores únicos de Place y Cause_of_Death
            Set<String> places = new HashSet<>();
            Set<String> generalCategories = new HashSet<>();

            // Extrae valores únicos de Place y Cause_of_Death
            for (int i = 1; i < csvData.size(); i++) {
                places.add(csvData.get(i)[4]); // Columna Place
                String specificLabel = csvData.get(i)[6].trim().toLowerCase(); // Columna Cause_of_Death
                String generalLabel = labelMapping.getOrDefault(specificLabel, specificLabel); // Usar el específico si no está en el mapeo

                // Formatear la causa de muerte eliminando espacios y caracteres conflictivos
                generalLabel = generalLabel.replaceAll("[^a-zA-Z0-9]", "_");
                generalCategories.add(generalLabel); // Añadirlo al conjunto
            }

            // Escribir el encabezado ARFF
            writeHeaderARFF(arffWriter, places, generalCategories);

            // Escribe los datos en el ARFF
            for (int i = 1; i < csvData.size(); i++) {
                String[] row = csvData.get(i);
                String specificLabel = row[6].trim().toLowerCase(); // Columna Cause_of_Death
                String generalLabel = labelMapping.getOrDefault(specificLabel, specificLabel); // Mapeo
                generalLabel = generalLabel.replaceAll("[^a-zA-Z0-9]", "_");

                row[6] = generalLabel; // Actualiza la causa de muerte
                row[5] = "\"" + cleanNarrative(row[5]) + "\""; // Limpia el texto

                // Escribe la fila en el ARFF
                arffWriter.write(String.join(",", row) + "\n");
            }

            // Crear los archivos
            arffWriter.close();
            reader.close();

            System.out.println("Archivo ARFF generado exitosamente en: " + arffFilePath);
        }
        catch (IOException e) {
            e.printStackTrace();
        }
        catch (CsvException e) {
            throw new RuntimeException(e);
        }

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
            fw.write("@ATTRIBUTE Age_Group {Adult, Child, Neonate}\n");
            fw.write("@ATTRIBUTE Age NUMERIC\n");
            fw.write("@ATTRIBUTE Sex {1, 2}\n");
            fw.write("@ATTRIBUTE Place {" + String.join(",", places) + "}\n");
            fw.write("@ATTRIBUTE Narrative STRING\n");
            fw.write("@ATTRIBUTE Cause_of_Death {" + String.join(",", generalCategories) + "}\n\n");
            fw.write("@DATA\n");
        }
        catch (IOException e) {
            e.printStackTrace();
        }
    }
}
