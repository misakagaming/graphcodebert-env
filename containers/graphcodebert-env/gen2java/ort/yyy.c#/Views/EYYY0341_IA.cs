// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
//
//                    Source Code Generated by
//                           CA Gen 8.6
//
//    Copyright (c) 2024 CA Technologies. All rights reserved.
//
//    Name: EYYY0341_IA                      Date: 2024/01/09
//    User: AliAl                            Time: 13:40:20
//
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// using Statements
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
using System;
using com.ca.gen.vwrt;
using com.ca.gen.vwrt.types;
using com.ca.gen.vwrt.vdf;
using com.ca.gen.csu.exception;

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// START OF IMPORT VIEW
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
namespace GEN.ORT.YYY
{
  /// <summary>
  /// Internal data view storage for: EYYY0341_IA
  /// </summary>
  [Serializable]
  public class EYYY0341_IA : ViewBase, IImportView
  {
    private static EYYY0341_IA[] freeArray = new EYYY0341_IA[30];
    private static int countFree = 0;
    
    // Entity View: IMP_REFERENCE
    //        Type: IYY1_SERVER_DATA
    /// <summary>
    /// Internal storage for attribute missing flag for: ImpReferenceIyy1ServerDataUserid
    /// </summary>
    private char _ImpReferenceIyy1ServerDataUserid_AS;
    /// <summary>
    /// Attribute missing flag for: ImpReferenceIyy1ServerDataUserid
    /// </summary>
    public char ImpReferenceIyy1ServerDataUserid_AS {
      get {
        return(_ImpReferenceIyy1ServerDataUserid_AS);
      }
      set {
        _ImpReferenceIyy1ServerDataUserid_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ImpReferenceIyy1ServerDataUserid
    /// Domain: Text
    /// Length: 8
    /// Varying Length: N
    /// </summary>
    private string _ImpReferenceIyy1ServerDataUserid;
    /// <summary>
    /// Attribute for: ImpReferenceIyy1ServerDataUserid
    /// </summary>
    public string ImpReferenceIyy1ServerDataUserid {
      get {
        return(_ImpReferenceIyy1ServerDataUserid);
      }
      set {
        _ImpReferenceIyy1ServerDataUserid = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ImpReferenceIyy1ServerDataReferenceId
    /// </summary>
    private char _ImpReferenceIyy1ServerDataReferenceId_AS;
    /// <summary>
    /// Attribute missing flag for: ImpReferenceIyy1ServerDataReferenceId
    /// </summary>
    public char ImpReferenceIyy1ServerDataReferenceId_AS {
      get {
        return(_ImpReferenceIyy1ServerDataReferenceId_AS);
      }
      set {
        _ImpReferenceIyy1ServerDataReferenceId_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ImpReferenceIyy1ServerDataReferenceId
    /// Domain: Timestamp
    /// Length: 20
    /// </summary>
    private string _ImpReferenceIyy1ServerDataReferenceId;
    /// <summary>
    /// Attribute for: ImpReferenceIyy1ServerDataReferenceId
    /// </summary>
    public string ImpReferenceIyy1ServerDataReferenceId {
      get {
        return(_ImpReferenceIyy1ServerDataReferenceId);
      }
      set {
        _ImpReferenceIyy1ServerDataReferenceId = value;
      }
    }
    // Entity View: IMP
    //        Type: TYPE
    /// <summary>
    /// Internal storage for attribute missing flag for: ImpTypeTinstanceId
    /// </summary>
    private char _ImpTypeTinstanceId_AS;
    /// <summary>
    /// Attribute missing flag for: ImpTypeTinstanceId
    /// </summary>
    public char ImpTypeTinstanceId_AS {
      get {
        return(_ImpTypeTinstanceId_AS);
      }
      set {
        _ImpTypeTinstanceId_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ImpTypeTinstanceId
    /// Domain: Timestamp
    /// Length: 20
    /// </summary>
    private string _ImpTypeTinstanceId;
    /// <summary>
    /// Attribute for: ImpTypeTinstanceId
    /// </summary>
    public string ImpTypeTinstanceId {
      get {
        return(_ImpTypeTinstanceId);
      }
      set {
        _ImpTypeTinstanceId = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ImpTypeTreferenceId
    /// </summary>
    private char _ImpTypeTreferenceId_AS;
    /// <summary>
    /// Attribute missing flag for: ImpTypeTreferenceId
    /// </summary>
    public char ImpTypeTreferenceId_AS {
      get {
        return(_ImpTypeTreferenceId_AS);
      }
      set {
        _ImpTypeTreferenceId_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ImpTypeTreferenceId
    /// Domain: Timestamp
    /// Length: 20
    /// </summary>
    private string _ImpTypeTreferenceId;
    /// <summary>
    /// Attribute for: ImpTypeTreferenceId
    /// </summary>
    public string ImpTypeTreferenceId {
      get {
        return(_ImpTypeTreferenceId);
      }
      set {
        _ImpTypeTreferenceId = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ImpTypeTkeyAttrText
    /// </summary>
    private char _ImpTypeTkeyAttrText_AS;
    /// <summary>
    /// Attribute missing flag for: ImpTypeTkeyAttrText
    /// </summary>
    public char ImpTypeTkeyAttrText_AS {
      get {
        return(_ImpTypeTkeyAttrText_AS);
      }
      set {
        _ImpTypeTkeyAttrText_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ImpTypeTkeyAttrText
    /// Domain: Text
    /// Length: 4
    /// Varying Length: N
    /// </summary>
    private string _ImpTypeTkeyAttrText;
    /// <summary>
    /// Attribute for: ImpTypeTkeyAttrText
    /// </summary>
    public string ImpTypeTkeyAttrText {
      get {
        return(_ImpTypeTkeyAttrText);
      }
      set {
        _ImpTypeTkeyAttrText = value;
      }
    }
    /// <summary>
    /// Default Constructor
    /// </summary>
    
    public EYYY0341_IA(  )
    {
      Reset(  );
    }
    /// <summary>
    /// Copy Constructor
    /// </summary>
    
    public EYYY0341_IA( EYYY0341_IA orig )
    {
      ImpReferenceIyy1ServerDataUserid_AS = orig.ImpReferenceIyy1ServerDataUserid_AS;
      ImpReferenceIyy1ServerDataUserid = orig.ImpReferenceIyy1ServerDataUserid;
      ImpReferenceIyy1ServerDataReferenceId_AS = orig.ImpReferenceIyy1ServerDataReferenceId_AS;
      ImpReferenceIyy1ServerDataReferenceId = orig.ImpReferenceIyy1ServerDataReferenceId;
      ImpTypeTinstanceId_AS = orig.ImpTypeTinstanceId_AS;
      ImpTypeTinstanceId = orig.ImpTypeTinstanceId;
      ImpTypeTreferenceId_AS = orig.ImpTypeTreferenceId_AS;
      ImpTypeTreferenceId = orig.ImpTypeTreferenceId;
      ImpTypeTkeyAttrText_AS = orig.ImpTypeTkeyAttrText_AS;
      ImpTypeTkeyAttrText = orig.ImpTypeTkeyAttrText;
    }
    /// <summary>
    /// Static instance creator function
    /// </summary>
    
    public static EYYY0341_IA GetInstance(  )
    {
      if ( countFree == 0 )
      {
        return(new EYYY0341_IA());
      }
      else 
      {
        lock (freeArray)
        {
          if ( countFree == 0 )
          {
            return(new EYYY0341_IA());
          }
          else 
          {
            EYYY0341_IA result = freeArray[--countFree];
            freeArray[countFree] = null;
            result.Reset(  );
            return(result);
          }
        }
      }
    }
    /// <summary>
    /// Static free instance method
    /// </summary>
    
    public void FreeInstance(  )
    {
      lock (freeArray)
      {
        if ( countFree < freeArray.Length )
        {
          freeArray[countFree++] = this;
        }
      }
    }
    /// <summary>
    /// clone constructor
    /// </summary>
    
    public override Object Clone(  )
    {
      return(new EYYY0341_IA(this));
    }
    /// <summary>
    /// Resets all properties to the defaults.
    /// </summary>
    
    public void Reset(  )
    {
      ImpReferenceIyy1ServerDataUserid_AS = ' ';
      ImpReferenceIyy1ServerDataUserid = "        ";
      ImpReferenceIyy1ServerDataReferenceId_AS = ' ';
      ImpReferenceIyy1ServerDataReferenceId = "00000000000000000000";
      ImpTypeTinstanceId_AS = ' ';
      ImpTypeTinstanceId = "00000000000000000000";
      ImpTypeTreferenceId_AS = ' ';
      ImpTypeTreferenceId = "00000000000000000000";
      ImpTypeTkeyAttrText_AS = ' ';
      ImpTypeTkeyAttrText = "    ";
    }
    /// <summary>
    /// Gets the VDF version of the current state of the instance.
    /// </summary>
    public VDF GetVDF(  )
    {
      throw new Exception("can only execute GetVDF for a Procedure Step.");
    }
    
    /// <summary>
    /// Sets the current state of the instance to the VDF version.
    /// </summary>
    public void SetFromVDF( VDF vdf )
    {
      throw new Exception("can only execute SetFromVDF for a Procedure Step.");
    }
    
    /// <summary>
    /// Sets the current instance based on the passed view.
    /// </summary>
    public void CopyFrom( IImportView orig )
    {
      this.CopyFrom((EYYY0341_IA) orig);
    }
    
    /// <summary>
    /// Sets the current instance based on the passed view.
    /// </summary>
    public void CopyFrom( EYYY0341_IA orig )
    {
      ImpReferenceIyy1ServerDataUserid_AS = orig.ImpReferenceIyy1ServerDataUserid_AS;
      ImpReferenceIyy1ServerDataUserid = orig.ImpReferenceIyy1ServerDataUserid;
      ImpReferenceIyy1ServerDataReferenceId_AS = orig.ImpReferenceIyy1ServerDataReferenceId_AS;
      ImpReferenceIyy1ServerDataReferenceId = orig.ImpReferenceIyy1ServerDataReferenceId;
      ImpTypeTinstanceId_AS = orig.ImpTypeTinstanceId_AS;
      ImpTypeTinstanceId = orig.ImpTypeTinstanceId;
      ImpTypeTreferenceId_AS = orig.ImpTypeTreferenceId_AS;
      ImpTypeTreferenceId = orig.ImpTypeTreferenceId;
      ImpTypeTkeyAttrText_AS = orig.ImpTypeTkeyAttrText_AS;
      ImpTypeTkeyAttrText = orig.ImpTypeTkeyAttrText;
    }
  }
}
