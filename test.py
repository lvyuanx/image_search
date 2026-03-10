import settings


GROUP_PATH_TEMPLATE = str(settings.BASE_DIR / "oss" / "media" / "image_search_workspaces" / "{group}")

print(GROUP_PATH_TEMPLATE.format(group="default"))