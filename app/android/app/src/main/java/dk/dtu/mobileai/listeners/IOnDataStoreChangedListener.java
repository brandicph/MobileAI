package dk.dtu.mobileai.listeners;

public interface IOnDataStoreChangedListener {

    void setOnDataStoreChangedListener(OnDataStoreChangedListener listener);

    void notifyDataChanged();

}
