// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
//
//                    Source Code Generated by
//                           CA Gen 8.6
//
//    Copyright (c) 2024 CA Technologies. All rights reserved.
//
//    Name: IYY10131_IA                      Date: 2024/01/09
//    User: AliAl                            Time: 13:41:56
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
  /// Internal data view storage for: IYY10131_IA
  /// </summary>
  [Serializable]
  public class IYY10131_IA : ViewBase, IImportView
  {
    private static IYY10131_IA[] freeArray = new IYY10131_IA[30];
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
    //        Type: IYY1_PARENT
    /// <summary>
    /// Internal storage for attribute missing flag for: ImpIyy1ParentPinstanceId
    /// </summary>
    private char _ImpIyy1ParentPinstanceId_AS;
    /// <summary>
    /// Attribute missing flag for: ImpIyy1ParentPinstanceId
    /// </summary>
    public char ImpIyy1ParentPinstanceId_AS {
      get {
        return(_ImpIyy1ParentPinstanceId_AS);
      }
      set {
        _ImpIyy1ParentPinstanceId_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ImpIyy1ParentPinstanceId
    /// Domain: Timestamp
    /// Length: 20
    /// </summary>
    private string _ImpIyy1ParentPinstanceId;
    /// <summary>
    /// Attribute for: ImpIyy1ParentPinstanceId
    /// </summary>
    public string ImpIyy1ParentPinstanceId {
      get {
        return(_ImpIyy1ParentPinstanceId);
      }
      set {
        _ImpIyy1ParentPinstanceId = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ImpIyy1ParentPreferenceId
    /// </summary>
    private char _ImpIyy1ParentPreferenceId_AS;
    /// <summary>
    /// Attribute missing flag for: ImpIyy1ParentPreferenceId
    /// </summary>
    public char ImpIyy1ParentPreferenceId_AS {
      get {
        return(_ImpIyy1ParentPreferenceId_AS);
      }
      set {
        _ImpIyy1ParentPreferenceId_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ImpIyy1ParentPreferenceId
    /// Domain: Timestamp
    /// Length: 20
    /// </summary>
    private string _ImpIyy1ParentPreferenceId;
    /// <summary>
    /// Attribute for: ImpIyy1ParentPreferenceId
    /// </summary>
    public string ImpIyy1ParentPreferenceId {
      get {
        return(_ImpIyy1ParentPreferenceId);
      }
      set {
        _ImpIyy1ParentPreferenceId = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ImpIyy1ParentPkeyAttrText
    /// </summary>
    private char _ImpIyy1ParentPkeyAttrText_AS;
    /// <summary>
    /// Attribute missing flag for: ImpIyy1ParentPkeyAttrText
    /// </summary>
    public char ImpIyy1ParentPkeyAttrText_AS {
      get {
        return(_ImpIyy1ParentPkeyAttrText_AS);
      }
      set {
        _ImpIyy1ParentPkeyAttrText_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ImpIyy1ParentPkeyAttrText
    /// Domain: Text
    /// Length: 5
    /// Varying Length: N
    /// </summary>
    private string _ImpIyy1ParentPkeyAttrText;
    /// <summary>
    /// Attribute for: ImpIyy1ParentPkeyAttrText
    /// </summary>
    public string ImpIyy1ParentPkeyAttrText {
      get {
        return(_ImpIyy1ParentPkeyAttrText);
      }
      set {
        _ImpIyy1ParentPkeyAttrText = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ImpIyy1ParentPsearchAttrText
    /// </summary>
    private char _ImpIyy1ParentPsearchAttrText_AS;
    /// <summary>
    /// Attribute missing flag for: ImpIyy1ParentPsearchAttrText
    /// </summary>
    public char ImpIyy1ParentPsearchAttrText_AS {
      get {
        return(_ImpIyy1ParentPsearchAttrText_AS);
      }
      set {
        _ImpIyy1ParentPsearchAttrText_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ImpIyy1ParentPsearchAttrText
    /// Domain: Text
    /// Length: 25
    /// Varying Length: N
    /// </summary>
    private string _ImpIyy1ParentPsearchAttrText;
    /// <summary>
    /// Attribute for: ImpIyy1ParentPsearchAttrText
    /// </summary>
    public string ImpIyy1ParentPsearchAttrText {
      get {
        return(_ImpIyy1ParentPsearchAttrText);
      }
      set {
        _ImpIyy1ParentPsearchAttrText = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ImpIyy1ParentPotherAttrText
    /// </summary>
    private char _ImpIyy1ParentPotherAttrText_AS;
    /// <summary>
    /// Attribute missing flag for: ImpIyy1ParentPotherAttrText
    /// </summary>
    public char ImpIyy1ParentPotherAttrText_AS {
      get {
        return(_ImpIyy1ParentPotherAttrText_AS);
      }
      set {
        _ImpIyy1ParentPotherAttrText_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ImpIyy1ParentPotherAttrText
    /// Domain: Text
    /// Length: 25
    /// Varying Length: N
    /// </summary>
    private string _ImpIyy1ParentPotherAttrText;
    /// <summary>
    /// Attribute for: ImpIyy1ParentPotherAttrText
    /// </summary>
    public string ImpIyy1ParentPotherAttrText {
      get {
        return(_ImpIyy1ParentPotherAttrText);
      }
      set {
        _ImpIyy1ParentPotherAttrText = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ImpIyy1ParentPtypeTkeyAttrText
    /// </summary>
    private char _ImpIyy1ParentPtypeTkeyAttrText_AS;
    /// <summary>
    /// Attribute missing flag for: ImpIyy1ParentPtypeTkeyAttrText
    /// </summary>
    public char ImpIyy1ParentPtypeTkeyAttrText_AS {
      get {
        return(_ImpIyy1ParentPtypeTkeyAttrText_AS);
      }
      set {
        _ImpIyy1ParentPtypeTkeyAttrText_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ImpIyy1ParentPtypeTkeyAttrText
    /// Domain: Text
    /// Length: 4
    /// Varying Length: N
    /// </summary>
    private string _ImpIyy1ParentPtypeTkeyAttrText;
    /// <summary>
    /// Attribute for: ImpIyy1ParentPtypeTkeyAttrText
    /// </summary>
    public string ImpIyy1ParentPtypeTkeyAttrText {
      get {
        return(_ImpIyy1ParentPtypeTkeyAttrText);
      }
      set {
        _ImpIyy1ParentPtypeTkeyAttrText = value;
      }
    }
    /// <summary>
    /// Default Constructor
    /// </summary>
    
    public IYY10131_IA(  )
    {
      Reset(  );
    }
    /// <summary>
    /// Copy Constructor
    /// </summary>
    
    public IYY10131_IA( IYY10131_IA orig )
    {
      ImpReferenceIyy1ServerDataUserid_AS = orig.ImpReferenceIyy1ServerDataUserid_AS;
      ImpReferenceIyy1ServerDataUserid = orig.ImpReferenceIyy1ServerDataUserid;
      ImpReferenceIyy1ServerDataReferenceId_AS = orig.ImpReferenceIyy1ServerDataReferenceId_AS;
      ImpReferenceIyy1ServerDataReferenceId = orig.ImpReferenceIyy1ServerDataReferenceId;
      ImpIyy1ParentPinstanceId_AS = orig.ImpIyy1ParentPinstanceId_AS;
      ImpIyy1ParentPinstanceId = orig.ImpIyy1ParentPinstanceId;
      ImpIyy1ParentPreferenceId_AS = orig.ImpIyy1ParentPreferenceId_AS;
      ImpIyy1ParentPreferenceId = orig.ImpIyy1ParentPreferenceId;
      ImpIyy1ParentPkeyAttrText_AS = orig.ImpIyy1ParentPkeyAttrText_AS;
      ImpIyy1ParentPkeyAttrText = orig.ImpIyy1ParentPkeyAttrText;
      ImpIyy1ParentPsearchAttrText_AS = orig.ImpIyy1ParentPsearchAttrText_AS;
      ImpIyy1ParentPsearchAttrText = orig.ImpIyy1ParentPsearchAttrText;
      ImpIyy1ParentPotherAttrText_AS = orig.ImpIyy1ParentPotherAttrText_AS;
      ImpIyy1ParentPotherAttrText = orig.ImpIyy1ParentPotherAttrText;
      ImpIyy1ParentPtypeTkeyAttrText_AS = orig.ImpIyy1ParentPtypeTkeyAttrText_AS;
      ImpIyy1ParentPtypeTkeyAttrText = orig.ImpIyy1ParentPtypeTkeyAttrText;
    }
    /// <summary>
    /// Static instance creator function
    /// </summary>
    
    public static IYY10131_IA GetInstance(  )
    {
      if ( countFree == 0 )
      {
        return(new IYY10131_IA());
      }
      else 
      {
        lock (freeArray)
        {
          if ( countFree == 0 )
          {
            return(new IYY10131_IA());
          }
          else 
          {
            IYY10131_IA result = freeArray[--countFree];
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
      return(new IYY10131_IA(this));
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
      ImpIyy1ParentPinstanceId_AS = ' ';
      ImpIyy1ParentPinstanceId = "00000000000000000000";
      ImpIyy1ParentPreferenceId_AS = ' ';
      ImpIyy1ParentPreferenceId = "00000000000000000000";
      ImpIyy1ParentPkeyAttrText_AS = ' ';
      ImpIyy1ParentPkeyAttrText = "     ";
      ImpIyy1ParentPsearchAttrText_AS = ' ';
      ImpIyy1ParentPsearchAttrText = "                         ";
      ImpIyy1ParentPotherAttrText_AS = ' ';
      ImpIyy1ParentPotherAttrText = "                         ";
      ImpIyy1ParentPtypeTkeyAttrText_AS = ' ';
      ImpIyy1ParentPtypeTkeyAttrText = "    ";
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
      this.CopyFrom((IYY10131_IA) orig);
    }
    
    /// <summary>
    /// Sets the current instance based on the passed view.
    /// </summary>
    public void CopyFrom( IYY10131_IA orig )
    {
      ImpReferenceIyy1ServerDataUserid_AS = orig.ImpReferenceIyy1ServerDataUserid_AS;
      ImpReferenceIyy1ServerDataUserid = orig.ImpReferenceIyy1ServerDataUserid;
      ImpReferenceIyy1ServerDataReferenceId_AS = orig.ImpReferenceIyy1ServerDataReferenceId_AS;
      ImpReferenceIyy1ServerDataReferenceId = orig.ImpReferenceIyy1ServerDataReferenceId;
      ImpIyy1ParentPinstanceId_AS = orig.ImpIyy1ParentPinstanceId_AS;
      ImpIyy1ParentPinstanceId = orig.ImpIyy1ParentPinstanceId;
      ImpIyy1ParentPreferenceId_AS = orig.ImpIyy1ParentPreferenceId_AS;
      ImpIyy1ParentPreferenceId = orig.ImpIyy1ParentPreferenceId;
      ImpIyy1ParentPkeyAttrText_AS = orig.ImpIyy1ParentPkeyAttrText_AS;
      ImpIyy1ParentPkeyAttrText = orig.ImpIyy1ParentPkeyAttrText;
      ImpIyy1ParentPsearchAttrText_AS = orig.ImpIyy1ParentPsearchAttrText_AS;
      ImpIyy1ParentPsearchAttrText = orig.ImpIyy1ParentPsearchAttrText;
      ImpIyy1ParentPotherAttrText_AS = orig.ImpIyy1ParentPotherAttrText_AS;
      ImpIyy1ParentPotherAttrText = orig.ImpIyy1ParentPotherAttrText;
      ImpIyy1ParentPtypeTkeyAttrText_AS = orig.ImpIyy1ParentPtypeTkeyAttrText_AS;
      ImpIyy1ParentPtypeTkeyAttrText = orig.ImpIyy1ParentPtypeTkeyAttrText;
    }
  }
}
