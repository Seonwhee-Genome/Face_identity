import os
import django
from collections import defaultdict

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'facetest.settings')
django.setup()

from vectorstore.models import Searchmanager

def deduplicate_searchmanager():
    # Change the fields here to define what constitutes a duplicate
    duplicate_key_fields = ['searchid']  

    # Use values_list to fetch just what we need, for performance
    records = Searchmanager.objects.values_list('id', *duplicate_key_fields).order_by('created_at')

    seen = set()
    duplicates_to_delete = []

    for record in records:
        record_id = record[0]
        key = tuple(record[1:])  # e.g., (user, searchid)

        if key in seen:
            duplicates_to_delete.append(record_id)
        else:
            seen.add(key)

    if duplicates_to_delete:        
        deleted_count, _ = Searchmanager.objects.filter(id__in=duplicates_to_delete).delete()
        print(f"Deleted {deleted_count} duplicate records.")
    else:
        print("No duplicates found.")

if __name__ == "__main__":
    deduplicate_searchmanager()
