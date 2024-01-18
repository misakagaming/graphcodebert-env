// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
//
//                    Source Code Generated by
//                           CA Gen 8.6
//
//    Copyright (c) 2024 CA Technologies. All rights reserved.
//
//    Name: IYY10231_IA                      Date: 2024/01/09
//    User: AliAl                            Time: 13:41:58
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
  /// Internal data view storage for: IYY10231_IA
  /// </summary>
  [Serializable]
  public class IYY10231_IA : ViewBase, IImportView
  {
    private static IYY10231_IA[] freeArray = new IYY10231_IA[30];
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
    //        Type: IYY1_CHILD
    /// <summary>
    /// Internal storage for attribute missing flag for: ImpIyy1ChildCinstanceId
    /// </summary>
    private char _ImpIyy1ChildCinstanceId_AS;
    /// <summary>
    /// Attribute missing flag for: ImpIyy1ChildCinstanceId
    /// </summary>
    public char ImpIyy1ChildCinstanceId_AS {
      get {
        return(_ImpIyy1ChildCinstanceId_AS);
      }
      set {
        _ImpIyy1ChildCinstanceId_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ImpIyy1ChildCinstanceId
    /// Domain: Timestamp
    /// Length: 20
    /// </summary>
    private string _ImpIyy1ChildCinstanceId;
    /// <summary>
    /// Attribute for: ImpIyy1ChildCinstanceId
    /// </summary>
    public string ImpIyy1ChildCinstanceId {
      get {
        return(_ImpIyy1ChildCinstanceId);
      }
      set {
        _ImpIyy1ChildCinstanceId = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ImpIyy1ChildCreferenceId
    /// </summary>
    private char _ImpIyy1ChildCreferenceId_AS;
    /// <summary>
    /// Attribute missing flag for: ImpIyy1ChildCreferenceId
    /// </summary>
    public char ImpIyy1ChildCreferenceId_AS {
      get {
        return(_ImpIyy1ChildCreferenceId_AS);
      }
      set {
        _ImpIyy1ChildCreferenceId_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ImpIyy1ChildCreferenceId
    /// Domain: Timestamp
    /// Length: 20
    /// </summary>
    private string _ImpIyy1ChildCreferenceId;
    /// <summary>
    /// Attribute for: ImpIyy1ChildCreferenceId
    /// </summary>
    public string ImpIyy1ChildCreferenceId {
      get {
        return(_ImpIyy1ChildCreferenceId);
      }
      set {
        _ImpIyy1ChildCreferenceId = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ImpIyy1ChildCparentPkeyAttrText
    /// </summary>
    private char _ImpIyy1ChildCparentPkeyAttrText_AS;
    /// <summary>
    /// Attribute missing flag for: ImpIyy1ChildCparentPkeyAttrText
    /// </summary>
    public char ImpIyy1ChildCparentPkeyAttrText_AS {
      get {
        return(_ImpIyy1ChildCparentPkeyAttrText_AS);
      }
      set {
        _ImpIyy1ChildCparentPkeyAttrText_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ImpIyy1ChildCparentPkeyAttrText
    /// Domain: Text
    /// Length: 5
    /// Varying Length: N
    /// </summary>
    private string _ImpIyy1ChildCparentPkeyAttrText;
    /// <summary>
    /// Attribute for: ImpIyy1ChildCparentPkeyAttrText
    /// </summary>
    public string ImpIyy1ChildCparentPkeyAttrText {
      get {
        return(_ImpIyy1ChildCparentPkeyAttrText);
      }
      set {
        _ImpIyy1ChildCparentPkeyAttrText = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ImpIyy1ChildCkeyAttrNum
    /// </summary>
    private char _ImpIyy1ChildCkeyAttrNum_AS;
    /// <summary>
    /// Attribute missing flag for: ImpIyy1ChildCkeyAttrNum
    /// </summary>
    public char ImpIyy1ChildCkeyAttrNum_AS {
      get {
        return(_ImpIyy1ChildCkeyAttrNum_AS);
      }
      set {
        _ImpIyy1ChildCkeyAttrNum_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ImpIyy1ChildCkeyAttrNum
    /// Domain: Number
    /// Length: 6
    /// Decimal Places: 0
    /// Decimal Precision: N
    /// </summary>
    private int _ImpIyy1ChildCkeyAttrNum;
    /// <summary>
    /// Attribute for: ImpIyy1ChildCkeyAttrNum
    /// </summary>
    public int ImpIyy1ChildCkeyAttrNum {
      get {
        return(_ImpIyy1ChildCkeyAttrNum);
      }
      set {
        _ImpIyy1ChildCkeyAttrNum = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ImpIyy1ChildCsearchAttrText
    /// </summary>
    private char _ImpIyy1ChildCsearchAttrText_AS;
    /// <summary>
    /// Attribute missing flag for: ImpIyy1ChildCsearchAttrText
    /// </summary>
    public char ImpIyy1ChildCsearchAttrText_AS {
      get {
        return(_ImpIyy1ChildCsearchAttrText_AS);
      }
      set {
        _ImpIyy1ChildCsearchAttrText_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ImpIyy1ChildCsearchAttrText
    /// Domain: Text
    /// Length: 25
    /// Varying Length: N
    /// </summary>
    private string _ImpIyy1ChildCsearchAttrText;
    /// <summary>
    /// Attribute for: ImpIyy1ChildCsearchAttrText
    /// </summary>
    public string ImpIyy1ChildCsearchAttrText {
      get {
        return(_ImpIyy1ChildCsearchAttrText);
      }
      set {
        _ImpIyy1ChildCsearchAttrText = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ImpIyy1ChildCotherAttrText
    /// </summary>
    private char _ImpIyy1ChildCotherAttrText_AS;
    /// <summary>
    /// Attribute missing flag for: ImpIyy1ChildCotherAttrText
    /// </summary>
    public char ImpIyy1ChildCotherAttrText_AS {
      get {
        return(_ImpIyy1ChildCotherAttrText_AS);
      }
      set {
        _ImpIyy1ChildCotherAttrText_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ImpIyy1ChildCotherAttrText
    /// Domain: Text
    /// Length: 25
    /// Varying Length: N
    /// </summary>
    private string _ImpIyy1ChildCotherAttrText;
    /// <summary>
    /// Attribute for: ImpIyy1ChildCotherAttrText
    /// </summary>
    public string ImpIyy1ChildCotherAttrText {
      get {
        return(_ImpIyy1ChildCotherAttrText);
      }
      set {
        _ImpIyy1ChildCotherAttrText = value;
      }
    }
    /// <summary>
    /// Default Constructor
    /// </summary>
    
    public IYY10231_IA(  )
    {
      Reset(  );
    }
    /// <summary>
    /// Copy Constructor
    /// </summary>
    
    public IYY10231_IA( IYY10231_IA orig )
    {
      ImpReferenceIyy1ServerDataUserid_AS = orig.ImpReferenceIyy1ServerDataUserid_AS;
      ImpReferenceIyy1ServerDataUserid = orig.ImpReferenceIyy1ServerDataUserid;
      ImpReferenceIyy1ServerDataReferenceId_AS = orig.ImpReferenceIyy1ServerDataReferenceId_AS;
      ImpReferenceIyy1ServerDataReferenceId = orig.ImpReferenceIyy1ServerDataReferenceId;
      ImpIyy1ChildCinstanceId_AS = orig.ImpIyy1ChildCinstanceId_AS;
      ImpIyy1ChildCinstanceId = orig.ImpIyy1ChildCinstanceId;
      ImpIyy1ChildCreferenceId_AS = orig.ImpIyy1ChildCreferenceId_AS;
      ImpIyy1ChildCreferenceId = orig.ImpIyy1ChildCreferenceId;
      ImpIyy1ChildCparentPkeyAttrText_AS = orig.ImpIyy1ChildCparentPkeyAttrText_AS;
      ImpIyy1ChildCparentPkeyAttrText = orig.ImpIyy1ChildCparentPkeyAttrText;
      ImpIyy1ChildCkeyAttrNum_AS = orig.ImpIyy1ChildCkeyAttrNum_AS;
      ImpIyy1ChildCkeyAttrNum = orig.ImpIyy1ChildCkeyAttrNum;
      ImpIyy1ChildCsearchAttrText_AS = orig.ImpIyy1ChildCsearchAttrText_AS;
      ImpIyy1ChildCsearchAttrText = orig.ImpIyy1ChildCsearchAttrText;
      ImpIyy1ChildCotherAttrText_AS = orig.ImpIyy1ChildCotherAttrText_AS;
      ImpIyy1ChildCotherAttrText = orig.ImpIyy1ChildCotherAttrText;
    }
    /// <summary>
    /// Static instance creator function
    /// </summary>
    
    public static IYY10231_IA GetInstance(  )
    {
      if ( countFree == 0 )
      {
        return(new IYY10231_IA());
      }
      else 
      {
        lock (freeArray)
        {
          if ( countFree == 0 )
          {
            return(new IYY10231_IA());
          }
          else 
          {
            IYY10231_IA result = freeArray[--countFree];
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
      return(new IYY10231_IA(this));
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
      ImpIyy1ChildCinstanceId_AS = ' ';
      ImpIyy1ChildCinstanceId = "00000000000000000000";
      ImpIyy1ChildCreferenceId_AS = ' ';
      ImpIyy1ChildCreferenceId = "00000000000000000000";
      ImpIyy1ChildCparentPkeyAttrText_AS = ' ';
      ImpIyy1ChildCparentPkeyAttrText = "     ";
      ImpIyy1ChildCkeyAttrNum_AS = ' ';
      ImpIyy1ChildCkeyAttrNum = 0;
      ImpIyy1ChildCsearchAttrText_AS = ' ';
      ImpIyy1ChildCsearchAttrText = "                         ";
      ImpIyy1ChildCotherAttrText_AS = ' ';
      ImpIyy1ChildCotherAttrText = "                         ";
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
      this.CopyFrom((IYY10231_IA) orig);
    }
    
    /// <summary>
    /// Sets the current instance based on the passed view.
    /// </summary>
    public void CopyFrom( IYY10231_IA orig )
    {
      ImpReferenceIyy1ServerDataUserid_AS = orig.ImpReferenceIyy1ServerDataUserid_AS;
      ImpReferenceIyy1ServerDataUserid = orig.ImpReferenceIyy1ServerDataUserid;
      ImpReferenceIyy1ServerDataReferenceId_AS = orig.ImpReferenceIyy1ServerDataReferenceId_AS;
      ImpReferenceIyy1ServerDataReferenceId = orig.ImpReferenceIyy1ServerDataReferenceId;
      ImpIyy1ChildCinstanceId_AS = orig.ImpIyy1ChildCinstanceId_AS;
      ImpIyy1ChildCinstanceId = orig.ImpIyy1ChildCinstanceId;
      ImpIyy1ChildCreferenceId_AS = orig.ImpIyy1ChildCreferenceId_AS;
      ImpIyy1ChildCreferenceId = orig.ImpIyy1ChildCreferenceId;
      ImpIyy1ChildCparentPkeyAttrText_AS = orig.ImpIyy1ChildCparentPkeyAttrText_AS;
      ImpIyy1ChildCparentPkeyAttrText = orig.ImpIyy1ChildCparentPkeyAttrText;
      ImpIyy1ChildCkeyAttrNum_AS = orig.ImpIyy1ChildCkeyAttrNum_AS;
      ImpIyy1ChildCkeyAttrNum = orig.ImpIyy1ChildCkeyAttrNum;
      ImpIyy1ChildCsearchAttrText_AS = orig.ImpIyy1ChildCsearchAttrText_AS;
      ImpIyy1ChildCsearchAttrText = orig.ImpIyy1ChildCsearchAttrText;
      ImpIyy1ChildCotherAttrText_AS = orig.ImpIyy1ChildCotherAttrText_AS;
      ImpIyy1ChildCotherAttrText = orig.ImpIyy1ChildCotherAttrText;
    }
  }
}
