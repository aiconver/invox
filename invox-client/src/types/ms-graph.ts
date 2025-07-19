export interface DiscoveredResource {
    type: 'onedrive' | 'sharepointLibrary';
    id: string; // driveId from Graph
    name: string;
    webUrl?: string;
    siteId?: string; // Only for SharePoint libraries
    siteName?: string; // Only for SharePoint libraries
    siteWebUrl?: string; // Only for SharePoint libraries
  }