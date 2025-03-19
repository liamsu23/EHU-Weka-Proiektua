import com.opencsv.CSVReader;
import com.opencsv.exceptions.CsvException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;
import java.util.HashMap;
import java.util.Map;

public class getARFF {

    public static void main(String[] args) {
        String csvFilePath = "hasierako_datuak_CSV/cleaned_PHMRC_VAI_redacted_free_text.train.csv"; // Ruta del archivo CSV
        String arffFilePath = "data/train_Zikina.arff"; // Ruta del archivo ARFF de salida
        String relationName = "muertes_causas"; // Nombre de la relación ARFF

        // Mapeo de etiquetas específicas a categorías generales
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

        try {
            // Leer el archivo CSV
            CSVReader reader = new CSVReader(new FileReader(csvFilePath));
            List<String[]> csvData = reader.readAll();

            // Crear el archivo ARFF
            FileWriter arffWriter = new FileWriter(arffFilePath);

            // Escribir el encabezado ARFF
            arffWriter.write("@RELATION " + relationName + "\n\n");

            // Definir los atributos con sus tipos correctos
            arffWriter.write("@ATTRIBUTE ID NUMERIC\n");
            arffWriter.write("@ATTRIBUTE Age_Group {Adult, Child, Neonate}\n");
            arffWriter.write("@ATTRIBUTE Age NUMERIC\n");
            arffWriter.write("@ATTRIBUTE Sex {1, 2}\n");

            // Obtener valores únicos para Place y definirlo como NOMINAL
            Set<String> places = new HashSet<>();
            for (int i = 1; i < csvData.size(); i++) {
                String[] row = csvData.get(i);
                places.add(row[4]); // Columna Place
            }
            arffWriter.write("@ATTRIBUTE Place {" + String.join(",", places) + "}\n");

            // Definir Narrative como STRING (se envolverá en comillas)
            arffWriter.write("@ATTRIBUTE Narrative STRING\n");

            // Obtener todas las categorías generales únicas para Cause_of_Death
            Set<String> generalCategories = new HashSet<>();

            for (int i = 1; i < csvData.size(); i++) {
                String specificLabel = csvData.get(i)[6].trim().toLowerCase(); // Convertir a minúsculas para evitar problemas
                String generalLabel = labelMapping.getOrDefault(specificLabel, specificLabel); // Usar el específico si no está en el mapeo

                // Formatear la causa de muerte eliminando espacios y caracteres conflictivos
                generalLabel = generalLabel.replaceAll("[^a-zA-Z0-9]", "_");

                generalCategories.add(generalLabel); // Añadirlo al conjunto
            }

            arffWriter.write("@ATTRIBUTE Cause_of_Death {" + String.join(",", generalCategories) + "}\n\n");

            // Escribir los datos
            arffWriter.write("@DATA\n");
            for (int i = 1; i < csvData.size(); i++) { // Empezar desde 1 para omitir el encabezado
                String[] row = csvData.get(i);
                String specificLabel = row[6].trim().toLowerCase(); // Convertir a minúsculas para normalizar
                String generalLabel = labelMapping.getOrDefault(specificLabel, specificLabel); // Usar el específico si no está en el mapeo

                // Formatear la etiqueta (sin espacios, sin caracteres conflictivos)
                generalLabel = generalLabel.replaceAll("[^a-zA-Z0-9]", "_");

                row[6] = generalLabel;

                // Limpiar el texto de Narrative
                String narrative = row[5]
                        .replace("#", "").replace("?", "").replace("!", "").replace(",", "")
                        .replace("\"", "").replace("'", "").replace(";", "").replace(":", "")
                        .replace("(", "").replace(")", "").replace("[", "").replace("]", "")
                        .replace("{", "").replace("}", "").replace("/", " ").replace("\\", " ")
                        .trim();

                // Envolver Narrative en comillas dobles
                row[5] = "\"" + narrative + "\"";

                // Escribir la fila en el archivo ARFF
                arffWriter.write(String.join(",", row) + "\n");
            }

            // Cerrar los recursos
            arffWriter.close();
            reader.close();

            System.out.println("Archivo ARFF generado exitosamente en: " + arffFilePath);

        } catch (IOException | CsvException e) {
            e.printStackTrace();
        }


    }
}
