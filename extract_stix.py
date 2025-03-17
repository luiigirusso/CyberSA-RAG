import stix2

def filter_stix_objects_by_techniques(stix_file_path, techniques, output_file_path):
    """
    Filtra gli oggetti STIX per una lista di tecniche e salva il sotto-grafo risultante in un file.
    
    Args:
        stix_file_path (str): Percorso del file STIX di input.
        techniques (list): Lista di nomi delle tecniche da filtrare.
        output_file_path (str): Percorso del file STIX filtrato in uscita.
    """
    # Carica il file STIX
    with open(stix_file_path, 'r') as f:
        stix_bundle = stix2.parse(f.read(), allow_custom=True)
    
    if not isinstance(stix_bundle, stix2.Bundle):
        raise ValueError("Il file fornito non contiene un oggetto STIX Bundle valido.")
    
    # Filtra gli oggetti STIX relativi alle tecniche desiderate
    filtered_objects = []
    for obj in stix_bundle.objects:
        if hasattr(obj, "name") and obj.name in techniques:
            filtered_objects.append(obj)
            # Trova relazioni e altri oggetti correlati
            related_objects = [
                rel for rel in stix_bundle.objects
                if hasattr(rel, "source_ref") and rel.source_ref == obj.id or
                   hasattr(rel, "target_ref") and rel.target_ref == obj.id
            ]
            filtered_objects.extend(related_objects)
    
    # Rimuovi duplicati
    filtered_objects = list({obj.id: obj for obj in filtered_objects}.values())
    
    # Crea un nuovo bundle con gli oggetti filtrati
    filtered_bundle = stix2.Bundle(objects=filtered_objects, allow_custom=True)
    
    # Salva il nuovo bundle in un file
    with open(output_file_path, 'w') as f:
        f.write(filtered_bundle.serialize(pretty=True))
    
    print(f"Sotto-grafo STIX salvato in: {output_file_path}")

# Percorso del file STIX di input
stix_file = "enterprise-attack.json"
# Lista delle tecniche desiderate
techniques = ["Reflection Amplification", "Filter Network Traffic", "Network Traffic", "Sensor Health"]
# Percorso per salvare il file STIX filtrato
output_file = "ReflectionAmplification.json"

# Filtra e salva il sotto-grafo
filter_stix_objects_by_techniques(stix_file, techniques, output_file)
